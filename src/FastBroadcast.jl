module FastBroadcast

export @..

using Base.Broadcast: Broadcasted
using LinearAlgebra: Adjoint, Transpose

getstyle(::Type{Broadcasted{S,Axes,F,Args}}) where {S,Axes,F,Args} = S
getAxes(::Type{Broadcasted{S,Axes,F,Args}}) where {S,Axes,F,Args} = Axes
getF(::Type{Broadcasted{S,Axes,F,Args}}) where {S,Axes,F,Args} = F
getArgs(::Type{Broadcasted{S,Axes,F,Args}}) where {S,Axes,F,Args} = collect(Args.parameters)
getAxes(::Type{T}) where {T<:Tuple} = collect(T.parameters)

use_fast_broadcast(_) = false
use_fast_broadcast(::Type{<:Base.Broadcast.DefaultArrayStyle}) = true
use_fast_broadcast(::Type{<:Base.Broadcast.DefaultArrayStyle{0}}) = false

@inline function fast_materialize(bc::Broadcasted{S}) where S
    if use_fast_broadcast(S)
        fast_materialize!(similar(bc, Base.Broadcast.combine_eltypes(bc.f, bc.args)), bc)
    else
        Base.Broadcast.materialize(bc)
    end
end

@inline function fast_materialize!(dst, bc::Broadcasted{S}) where S
    if use_fast_broadcast(S)
        fast_materialize!(dst, bc, axes(dst), _get_axes(bc), _index_style(bc))
    else
        Base.Broadcast.materialize!(dst, bc)
    end
end

@generated function fast_materialize!(dst, bc::Broadcasted, dstaxes::Tuple{Vararg{Any,N}}, ax, indexstyle) where {N}
    loopbody_lin = :($setindex!(dst))
    loopbody_car = :($setindex!(dst))
    bcc = BroadcastCharacteristics()
    ii = map(i->Symbol(:i_, i), 1:N)

    walk_bc!(
        bcc, loopbody_lin, loopbody_car,
        ii, bc, :bc, ax, :ax
       )
    push!(loopbody_lin.args, :i)
    append!(loopbody_car.args, ii)

    loop_quote = if !(bcc.maybelinear && (indexstyle === IndexLinear))
        :(@inbounds Base.Cartesian.@nloops $N i dst begin
            $loopbody_car
        end)
    elseif bcc.maybeivdep
        :(@inbounds @simd ivdep for i in $eachindex(dst)
            $loopbody_lin
        end)
    else
        :(@inbounds @simd for i in $eachindex(dst)
            $loopbody_lin
        end)
    end
    quote
        $(Expr(:meta,:inline))
        isfast = true
        (Base.Cartesian.@ntuple $N dstaxis) = dstaxes
        $(bcc.loopheader)
        if isfast
            $loop_quote
        else
            slow_materialize!(dst, bc)
        end
        dst
    end
end
@inline _get_axes(x) = axes(x)
@inline _get_axes(bc::Broadcasted) = map(_get_axes, bc.args)
@inline __index_style(_) = Val{false}()
@inline __index_style(::IndexLinear) = Val{true}()

# Two arg `_index_style` checks, either stopping with `Val{false}()`, or continues
@inline _index_style(_, __) = IndexCartesian()
@inline _index_style(::IndexLinear, x) = _index_style(x)
@inline _index_style(::IndexLinear, x::Tuple{}) = IndexLinear()
@inline _index_style(::IndexLinear, x::Tuple{T}) where {T} = _index_style(first(x))
@inline _index_style(::IndexLinear, x::Tuple{T,S,Vararg}) where {T,S} = _index_style(_index_style(first(x)), Base.tail(x))

@inline _index_style(x) = IndexStyle(typeof(x)) # require `IndexStyle` to be defined
@inline _index_style(x::Tuple) = IndexLinear()
@inline _index_style(x::Number) = IndexLinear()
@inline _index_style(x::Ref) = IndexLinear()
@inline _index_style(x::AbstractArray) = IndexStyle(x)

@inline _index_style(bc::Broadcasted) = _index_style(_index_style(first(bc.args)), Base.tail(bc.args))


@noinline slow_materialize!(dest, bc) = Base.Broadcast.materialize!(dest, bc)

fast_materialize!(dest, x::Number) = fill!(dest, x)
fast_materialize!(dest, x::AbstractArray) = copyto!(dest, x)

safeivdep(_) = false
safeivdep(::Type{Array{T,N}}) where {T <: Union{Bool,Base.HWNumber},N} = true
safeivdep(::Type{Adjoint{T,Array{T,N}}}) where {T <: Union{Bool,Base.HWNumber}, N} = true
safeivdep(::Type{Transpose{T,Array{T,N}}}) where {T <: Union{Bool,Base.HWNumber}, N} = true
safeivdep(::Type{SubArray{T,N,Array{T,M}}}) where {T <: Union{Bool,Base.HWNumber}, N, M} = true

mutable struct BroadcastCharacteristics
    loopheader::Expr
    arrays::Vector{Symbol}
    maybelinear::Bool
    maybeivdep::Bool
end
BroadcastCharacteristics() = BroadcastCharacteristics(Expr(:block), Symbol[], true, true)

_tuplelen(::Type{T}) where {N,T<:Tuple{Vararg{Any,N}}} = N

function walk_bc!(
        bcc::BroadcastCharacteristics, loopbody_lin, loopbody_car,
        ii, bc::Type{<:Broadcasted}, bcsym, ax::Type{<:Tuple}, axsym
       )
    f = gensym(:f)
    push!(bcc.loopheader.args, :($f = $bcsym.f))
    new_loopbody_lin = Expr(:call, f)
    new_loopbody_car = Expr(:call, f)
    args = getArgs(bc)
    axs  = getAxes(ax)
    push!(loopbody_lin.args, new_loopbody_lin)
    push!(loopbody_car.args, new_loopbody_car)
    for (i, arg) in enumerate(args)
        if arg <: Broadcasted
            new_bcsym = gensym(:bcsym); new_axsym = gensym(:axsym);
            push!(bcc.loopheader.args, :($new_bcsym = $bcsym.args[$i]))
            push!(bcc.loopheader.args, :($new_axsym = $axsym[$i]))
            walk_bc!(bcc, new_loopbody_lin, new_loopbody_car, ii, arg, new_bcsym, axs[i], new_axsym)
        else
            new_arg = gensym(:x)
            push!(bcc.loopheader.args, :($new_arg = $bcsym.args[$i]))
            nd::Int = length(ii)
            if (arg <: Adjoint{<:Any,<:AbstractVector}) || (arg <: Transpose{<:Any,<:AbstractVector})
                push!(bcc.arrays, new_arg)
                bcc.maybelinear = false
                new_arg_parent = Symbol(new_arg, "##parent##")
                push!(bcc.loopheader.args, :($new_arg_parent = parent($new_arg)))
                push!(bcc.loopheader.args, :(isfast &= axes($new_arg_parent,1) == dstaxis_2))
                index = :($new_arg_parent[$(ii[2])])
                if eltype(arg) <: Base.HWReal
                    nothing # `adjoint` and `transpose` are the identity
                elseif (arg <: Adjoint)
                    index = :(adjoint($index))
                else
                    index = :(transpose($index))
                end
                push!(new_loopbody_car.args, index)
            elseif arg <: Tuple
                tuple_length = _tuplelen(arg)
                if tuple_length == 1
                  scalar = gensym(:scalar)
                  push!(bcc.loopheader.args, :($scalar = $new_arg[1]))
                  push!(new_loopbody_lin.args, scalar)
                  push!(new_loopbody_car.args, scalar)
                else
                  bcc.maybelinear &= nd == 1
                  push!(bcc.loopheader.args, :(isfast &= Base.OneTo($tuple_length) == dstaxis_1))
                  push!(new_loopbody_lin.args, :($new_arg[i]))
                  push!(new_loopbody_car.args, :($new_arg[i1]))
                end
            else
                new_nd::Int = _tuplelen(axs[i]) # ndims on `arg` won't work because of possible world age errors.
                if new_nd == 0
                    scalar = gensym(:scalar)
                    push!(bcc.loopheader.args, :($scalar = $new_arg[]))
                    push!(new_loopbody_lin.args, scalar)
                    push!(new_loopbody_car.args, scalar)
                else
                    push!(bcc.arrays, new_arg)
                    bcc.maybelinear &= (nd == new_nd)
                    bcc.maybeivdep = bcc.maybeivdep && safeivdep(arg)
                    new_arg_axes = Symbol(new_arg, "#axes#")
                    push!(bcc.loopheader.args, :($new_arg_axes = $axsym[$i]))
                    push!(bcc.loopheader.args, :((Base.Cartesian.@ntuple $new_nd $new_arg_axes) = $new_arg_axes))
                    for n ∈ 1:new_nd
                        push!(bcc.loopheader.args, :(isfast &= $(Symbol(new_arg_axes,'_',n)) == $(Symbol(:dstaxis_,n))))
                    end
                    push!(new_loopbody_lin.args, :($new_arg[i]))
                    push!(new_loopbody_car.args, :($new_arg[$(ii[1:new_nd]...)]))
                end
            end
        end
    end
    return nothing
end

# From Julia Base
dottable(x) = false # avoid dotting spliced objects (e.g. view calls inserted by @view)
# don't add dots to dot operators
dottable(x::Symbol) = (!Base.isoperator(x) || first(string(x)) != '.' || x === :..) && x !== :(:)
dottable(x::Expr) = x.head !== :$

function broadcasted_expr!(_ex)
    (Meta.isexpr(_ex,:call) && dottable(_ex.args[1])) || return _ex
    ex::Expr = _ex
    call = Expr(:call, Base.Broadcast.broadcasted, ex.args[1])
    for n ∈ 2:length(ex.args)
        push!(call.args, broadcasted_expr!(ex.args[n]))
    end
    call
end

function broadcast_expr!(ex::Expr)
    update = findfirst(isequal(ex.head), (:(+=), :(-=), :(*=), :(/=), :(\=), :(^=), :(&=), :(|=), :(⊻=), :(÷=)))
    if update ≢ nothing
        lhs = Expr(:call, (:(+), :(-), :(*), :(/), :(\), :(^), :(&), :(|), :(⊻), :(÷))[update], ex.args[1], ex.args[2])
        ex = Expr(:(=), ex.args[1], lhs)
    end
    if Meta.isexpr(ex, :(=), 2)
        return Expr(:call, fast_materialize!, ex.args[1], broadcasted_expr!(ex.args[2]))
    else
        return Expr(:call, fast_materialize, broadcasted_expr!(ex))
    end
end

macro (..)(ex)
    esc(broadcast_expr!(ex))
end

end
