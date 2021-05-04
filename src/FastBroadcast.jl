module FastBroadcast

export fast_materialize!, @..

using Base.Broadcast: Broadcasted

getstyle(::Type{Broadcasted{S,Axes,F,Args}}) where {S,Axes,F,Args} = S
getAxes(::Type{Broadcasted{S,Axes,F,Args}}) where {S,Axes,F,Args} = Axes
getF(::Type{Broadcasted{S,Axes,F,Args}}) where {S,Axes,F,Args} = F
getArgs(::Type{Broadcasted{S,Axes,F,Args}}) where {S,Axes,F,Args} = collect(Args.parameters)

@generated function fast_materialize!(dst::AbstractArray, bc::Broadcasted)
    loopbody_lin = :($setindex!(dst))
    loopbody_car = :($setindex!(dst))
    bcc = BroadcastCharacteristics(true, Expr(:block), [])
    ii = map(i->Symbol(:i_, i), 1:ndims(dst))

    walk_bc!(
        bcc, loopbody_lin, loopbody_car,
        ii, bc, :bc,
       )
    push!(loopbody_lin.args, :i)
    append!(loopbody_car.args, ii)

    quote
        $(Expr(:meta,:inline))
        $(bcc.loopheader)
        if $islinear($axes(dst), $(bcc.arrays...))
            if $safeivdep(dst, $(bcc.arrays...))
                @inbounds @simd ivdep for i in $eachindex(dst)
                    $loopbody_lin
                end
            else
                @inbounds @simd for i in $eachindex(dst)
                    $loopbody_lin
                end
            end
        else
            @inbounds Base.Cartesian.@nloops $(ndims(dst)) i dst begin
                $loopbody_car
            end
        end
        dst
    end
end
@inline fast_materialize(bc::Broadcasted) = fast_materialize!(similar(bc, Base.Broadcast.combine_eltypes(bc.f, bc.args)), bc)

@inline islinear(dst, src) = dst == axes(src)
@inline islinear(dst, src, srcs::Vararg{AbstractArray,K}) where {K} = dst == axes(src) && islinear(dst, srcs...)

@inline safeivdep(::Array{T}) where {T<:Union{Bool,Base.HWReal}} = true
@inline safeivdep(::Array{T}, arg1, args::Vararg{Array,K}) where {K,T<:Union{Bool,Base.HWReal}} = safeivdep(arg1, args...)
@inline safeivdep(args::Vararg{Any,K}) where {K} = false

mutable struct BroadcastCharacteristics
    maybelinear::Bool
    loopheader::Expr
    arrays::Vector
end

function walk_bc!(
        bcc::BroadcastCharacteristics, loopbody_lin, loopbody_car,
        ii, bc::Type{<:Broadcasted}, bcsym,
       )
    f = gensym(:f)
    push!(bcc.loopheader.args, :($f = $bcsym.f))
    new_loopbody_lin = Expr(:call, f)
    new_loopbody_car = Expr(:call, f)
    args = getArgs(bc)
    push!(loopbody_lin.args, new_loopbody_lin)
    push!(loopbody_car.args, new_loopbody_car)
    for (i, arg) in enumerate(args)
        if arg <: Broadcasted
            new_bcsym = gensym(:bcsym)
            push!(bcc.loopheader.args, :($new_bcsym = $bcsym.args[$i]))
            walk_bc!(bcc, new_loopbody_lin, new_loopbody_car, ii, arg, new_bcsym)
        else
            new_arg = gensym(:x)
            push!(bcc.loopheader.args, :($new_arg = $bcsym.args[$i]))
            nd = length(ii)
            if arg <: AbstractArray
                push!(bcc.arrays, new_arg)
                new_nd = ndims(arg)
                bcc.maybelinear &= nd == new_nd
                push!(new_loopbody_lin.args, :($new_arg[i]))
                push!(new_loopbody_car.args, :($new_arg[$(ii[1:new_nd]...)]))
            elseif arg <: Tuple
                new_nd = 1
                bcc.maybelinear &= nd == new_nd
                push!(new_loopbody_lin.args, :($new_arg[i]))
                push!(new_loopbody_car.args, :($new_arg[i1]))
            else # ndims(arg) == 0
                push!(new_loopbody_lin.args, :($new_arg[]))
                push!(new_loopbody_car.args, :($new_arg[]))
            end
        end
    end
    return nothing
end

function broadcasted_expr!(_ex)
  Meta.isexpr(_ex,:call) || return _ex
  ex::Expr = _ex
  t = Expr(:tuple)
  for n ∈ 2:length(ex.args)
    push!(t.args, broadcasted_expr!(ex.args[n]))
  end
  Expr(:call, GlobalRef(Broadcast, :Broadcasted), ex.args[1], t)
end
function broadcast_expr!(ex::Expr)
  if Meta.isexpr(ex, :(=), 2)
    return Expr(:call, GlobalRef(FastBroadcast, :fast_materialize!), ex.args[1], broadcasted_expr!(ex.args[2]))
  else
    return Expr(:call, GlobalRef(FastBroadcast, :fast_materialize), broadcasted_expr!(ex))
  end
end
macro var".."(ex)
  esc(broadcast_expr!(ex))
end

end
