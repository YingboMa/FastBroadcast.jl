module FastBroadcast

export @..

using StaticArrayInterface: static_axes, known_length
using ArrayInterface: indices_do_not_alias
using Base.Broadcast: Broadcasted
using LinearAlgebra: Adjoint, Transpose
using Static, Polyester
using StrideArraysCore: AbstractStrideArray

getstyle(::Type{Broadcasted{S,Axes,F,Args}}) where {S,Axes,F,Args} = S
getAxes(::Type{Broadcasted{S,Axes,F,Args}}) where {S,Axes,F,Args} = Axes
getF(::Type{Broadcasted{S,Axes,F,Args}}) where {S,Axes,F,Args} = F
getArgs(::Type{Broadcasted{S,Axes,F,Args}}) where {S,Axes,F,Args} = collect(Args.parameters)
getAxes(::Type{T}) where {T<:Tuple} = collect(T.parameters)

use_fast_broadcast(_) = false
use_fast_broadcast(::Type{<:Base.Broadcast.DefaultArrayStyle}) = true
use_fast_broadcast(::Type{<:Base.Broadcast.DefaultArrayStyle{0}}) = false

@inline function fast_materialize(::SB, ::DB, bc::Broadcasted{S}) where {S,SB,DB}
    if use_fast_broadcast(S)
        fast_materialize!(
            SB(),
            DB(),
            similar(bc, Base.Broadcast.combine_eltypes(bc.f, bc.args)),
            bc,
        )
    else
        Base.Broadcast.materialize(bc)
    end
end

@inline function fast_materialize!(::False, ::DB, dst::A, bc::Broadcasted{S}) where {S,DB,A}
    if use_fast_broadcast(S)
        _fast_materialize!(
            dst,
            Val(indices_do_not_alias(A)),
            DB(),
            bc,
            static_axes(dst),
            _get_axes(bc),
            _index_style(bc),
        )
    else
        Base.Broadcast.materialize!(dst, bc)
    end
end

_view(A::AbstractArray{<:Any,N}, r, ::Val{N}) where {N} =
    view(A, ntuple(_ -> :, N - 1)..., r)
_view(A::AbstractArray, r, ::Val) = A
_view(x, r, ::Val) = x
__view(t::Tuple{T}, r, ::Val{N}) where {T,N} = (_view(first(t), r, Val(N)),)
__view(t::Tuple{T,Vararg}, r, ::Val{N}) where {T,N} =
    (_view(first(t), r, Val(N)), __view(Base.tail(t), r, Val(N))...)
function _view(
    bc::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{N},Nothing},
    r,
    ::Val{N},
) where {N}
    Base.Broadcast.Broadcasted(bc.f, __view(bc.args, r, Val(N)), Val(N))
end
_view(
    bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle},
    r,
    ::Val{N},
) where {N} = bc
_view(t::Tuple{Vararg{AbstractRange,N}}, r, ::Val{N}) where {N} = (Base.front(t)..., r)

function fast_materialize!(::True, ::DB, dst, bc::Broadcasted{S}) where {S,DB}
    if use_fast_broadcast(S)
        fast_materialize_threaded!(dst, DB(), bc, static_axes(dst))
    else
        Base.Broadcast.materialize!(dst, bc)
    end
end
@inline function _batch_broadcast_fn((dest, ldstaxes, bcobj, VN, DoBroadcast), start, stop)
    r = @inbounds ldstaxes[start:stop]
    fast_materialize!(False(), DoBroadcast, _view(dest, r, VN), _view(bcobj, r, VN))
    return nothing
end
function fast_materialize_threaded!(
    dst,
    ::DB,
    bc::Broadcasted,
    dstaxes::Tuple{Vararg{Any,N}},
) where {N,DB}
    last_dstaxes = dstaxes[N]
    Polyester.batch(
        _batch_broadcast_fn,
        (length(last_dstaxes), Threads.nthreads()),
        dst,
        last_dstaxes,
        bc,
        Val(N),
        DB(),
    )
    return dst
end


@inline _get_axes(x) = static_axes(x)
@inline _get_axes(bc::Broadcasted) = map(_get_axes, bc.args)
@inline __index_style(_) = Val{false}()
@inline __index_style(::IndexLinear) = Val{true}()

# Two arg `_index_style` checks, either stopping with `Val{false}()`, or continues
@inline _index_style(_, __) = IndexCartesian()
@inline _index_style(::IndexLinear, x) = _index_style(x)
@inline _index_style(::IndexLinear, x::Tuple{}) = IndexLinear()
@inline _index_style(::IndexLinear, x::Tuple{T}) where {T} = _index_style(first(x))
@inline _index_style(::IndexLinear, x::Tuple{T,S,Vararg}) where {T,S} =
    _index_style(_index_style(first(x)), Base.tail(x))

@inline _index_style(x) = IndexStyle(typeof(x)) # require `IndexStyle` to be defined
@inline _index_style(x::Tuple) = IndexLinear()
@inline _index_style(x::Number) = IndexLinear()
@inline _index_style(x::Ref) = IndexLinear()
@inline _index_style(x::AbstractArray) = IndexStyle(x)

@inline _index_style(bc::Broadcasted) =
    _index_style(_index_style(first(bc.args)), Base.tail(bc.args))

@generated function broadcastgetindex(A, i::Vararg{Any,N}) where {N}
    quote
        $(Expr(:meta, :inline))
        Base.Cartesian.@nref $N A n ->
            ifelse(size(A, n) == 1, (firstindex(A, n) % Int)::Int, (i[n] % Int)::Int)
    end
end

fast_materialize!(_, __, dest, x::Number) = fill!(dest, x)
fast_materialize!(_, __, dest, x::AbstractArray) =
    length(x) == 1 ? fill!(dest, x) : copyto!(dest, x)

mutable struct BroadcastCharacteristics
    loopheader::Expr
    arrays::Vector{Symbol}
    maybelinear::Bool
end
BroadcastCharacteristics() = BroadcastCharacteristics(Expr(:block), Symbol[], true)

_tuplelen(::Type{T}) where {N,T<:Tuple{Vararg{Any,N}}} = N

function walk_bc!(
    bcc::BroadcastCharacteristics,
    loopbody_lin::Expr,
    loopbody_car::Expr,
    loopbody_slow::Expr,
    ii::Vector{Symbol},
    @nospecialize(bc::Type{<:Broadcasted}),
    bcsym::Symbol,
    @nospecialize(ax::Type{<:Tuple}),
    axsym::Symbol,
)
    f = gensym(:f)
    push!(bcc.loopheader.args, :($f = $bcsym.f))
    new_loopbody_lin = Expr(:call, f)
    new_loopbody_car = Expr(:call, f)
    new_loopbody_slow = Expr(:call, f)
    args = getArgs(bc)
    axs = getAxes(ax)
    push!(loopbody_lin.args, new_loopbody_lin)
    push!(loopbody_car.args, new_loopbody_car)
    push!(loopbody_slow.args, new_loopbody_slow)
    for (i, arg) in enumerate(args)
        if arg <: Broadcasted
            new_bcsym = gensym(:bcsym)
            new_axsym = gensym(:axsym)
            push!(bcc.loopheader.args, :($new_bcsym = $bcsym.args[$i]))
            push!(bcc.loopheader.args, :($new_axsym = $axsym[$i]))
            walk_bc!(
                bcc,
                new_loopbody_lin,
                new_loopbody_car,
                new_loopbody_slow,
                ii,
                arg,
                new_bcsym,
                axs[i],
                new_axsym,
            )
        else
            new_arg = gensym(:x)
            push!(bcc.loopheader.args, :($new_arg = $bcsym.args[$i]))
            nd::Int = length(ii)
            if arg <: Tuple
                tuple_length = _tuplelen(arg)
                if tuple_length == 1
                    scalar = gensym(:scalar)
                    push!(bcc.loopheader.args, :($scalar = $new_arg[1]))
                    push!(new_loopbody_lin.args, scalar)
                    push!(new_loopbody_car.args, scalar)
                    push!(new_loopbody_slow.args, scalar)
                else
                    bcc.maybelinear &= nd == 1
                    push!(
                        bcc.loopheader.args,
                        :(isfast &= Base.OneTo($tuple_length) == dstaxis_1),
                    )
                    push!(new_loopbody_lin.args, :($new_arg[i]))
                    push!(new_loopbody_car.args, :($new_arg[i1]))
                    push!(new_loopbody_slow.args, :($new_arg[i1]))
                end
            else
                new_nd::Int = _tuplelen(axs[i]) # ndims on `arg` won't work because of possible world age errors.
                if new_nd == 0
                    scalar = gensym(:scalar)
                    push!(bcc.loopheader.args, :($scalar = $new_arg[]))
                    push!(new_loopbody_lin.args, scalar)
                    push!(new_loopbody_car.args, scalar)
                    push!(new_loopbody_slow.args, scalar)
                else
                    push!(bcc.arrays, new_arg)
                    bcc.maybelinear &= (nd == new_nd)
                    new_arg_axes = Symbol(new_arg, "#axes#")
                    push!(bcc.loopheader.args, :($new_arg_axes = $axsym[$i]))
                    push!(
                        bcc.loopheader.args,
                        :((Base.Cartesian.@ntuple $new_nd $new_arg_axes) = $new_arg_axes),
                    )
                    newarg_cartesian_index = Expr(:ref, new_arg)
                    newarg_slow_index = Expr(:call, broadcastgetindex, new_arg)
                    for n ∈ 1:new_nd
                        kl = known_length(axs[i].parameters[n])
                        if kl === 1
                            bcc.maybelinear = false
                            push!(newarg_cartesian_index.args, 1)
                            push!(newarg_slow_index.args, 1)
                        else
                            push!(
                                bcc.loopheader.args,
                                :(
                                    isfast &=
                                        $(Symbol(new_arg_axes, '_', n)) ==
                                        $(Symbol(:dstaxis_, n))
                                ),
                            )
                            push!(newarg_cartesian_index.args, ii[n])
                            push!(newarg_slow_index.args, ii[n])
                        end
                    end
                    push!(new_loopbody_lin.args, :($new_arg[i]))
                    push!(new_loopbody_car.args, newarg_cartesian_index)
                    push!(
                        new_loopbody_slow.args,
                        :(broadcastgetindex($new_arg, $(ii[1:new_nd]...))),
                    )
                end
            end
        end
    end
    return nothing
end

function pushsymname!(ex::Expr, base::Symbol, @nospecialize(arg))
    if arg isa Core.SSAValue
        push!(ex.args, Symbol(base, '_', arg.id))
    elseif arg isa Core.SlotNumber
        push!(ex.args, Symbol(base, 's', arg.id))
    else
        push!(ex.args, arg)
    end
end
function _goto(base::Symbol, i::Int, sym::Symbol)
    Expr(
        :macrocall,
        sym,
        LineNumberNode(@__LINE__, Symbol(@__FILE__)),
        Symbol(base, "#label#", i),
    )
end
goto(base::Symbol, i::Int) = _goto(base, i, Symbol("@goto"))
label(base::Symbol, i::Int) = _goto(base, i, Symbol("@label"))

function add_gotoifnot!(q::Expr, gotos::Vector{Int}, base::Symbol, cond, dest::Int)
    ex = Expr(:||)
    pushsymname!(ex, base, cond)
    push!(ex.args, goto(base, dest))
    push!(q.args, ex)
    push!(gotos, dest)
    nothing
end

function broadcast_codeinfo(ci, threadarg, broadcastarg)
    q = Expr(:block)
    base = gensym(:fastbroadcast)
    gotos = Int[]
    for (i, code) ∈ enumerate(ci.code)
        k = findfirst(==(i), gotos)
        if k ≢ nothing
            push!(q.args, label(base, i))
        end
        if Meta.isexpr(code, :call)
            ex = Expr(:call)
            f = code.args[1]
            if f === GlobalRef(Base, :materialize)
                push!(ex.args, fast_materialize, threadarg, broadcastarg)
            elseif f === GlobalRef(Base, :materialize!)
                push!(ex.args, fast_materialize!, threadarg, broadcastarg)
            elseif f === GlobalRef(Base, :getindex)
                push!(ex.args, Base.Broadcast.dotview)
            else
                pushsymname!(ex, base, f)
            end
            for arg ∈ @view(code.args[2:end])
                pushsymname!(ex, base, arg)
            end
            push!(q.args, Expr(:(=), Symbol(base, '_', i), ex))
        elseif Meta.isexpr(code, :(=))
            ex = Expr(:(=), Symbol(base, 's', code.args[1].id))
            pushsymname!(ex, base, code.args[2])
            push!(q.args, ex)
        elseif VERSION ≥ v"1.6" && code isa Core.GotoIfNot
            add_gotoifnot!(q, gotos, base, code.cond, code.dest)
        elseif VERSION < v"1.6" && Meta.isexpr(code, :gotoifnot)
            add_gotoifnot!(q, gotos, base, code.args[1], code.args[2])
        elseif code isa Core.GotoNode
            push!(q.args, goto(base, code.label))
            push!(gotos, code.label)
        elseif !(VERSION ≥ v"1.6" ? isa(code, Core.ReturnNode) : Meta.isexpr(code, :return))
            ex = Expr(:(=), Symbol(base, '_', i))
            pushsymname!(ex, base, code)
            push!(q.args, ex)
        end
    end
    q
end

function fb_macro(ex, mod, threadarg, broadcastarg)
    lowered = Meta.lower(mod, Base.Broadcast.__dot__(ex))
    lowered isa Expr || return esc(lowered)
    esc(broadcast_codeinfo(lowered.args[1], threadarg, broadcastarg))
end

"""
    @.. [thread=false] [broadcast=false] expr

`@..` turns `expr` into a broadcast-like expression, similar to `@.`.
It additionally provides two optional keyword arguments:

- thread: Defaults to `false`. Use multithreading?
- broadcast: Defaults to `false`. If `true`, it will broadcast axes with dynamic
    runtime sizes of `1` to larger sizes, if `false` only sizes known to be `1`
    at compile time will be supported, i.e. axes such that
    `ArrayInterface.known_length(typeof(StaticArrayInterface.static_axes(x,i))) == 1` will
    be broadcast. Note that this differs from base broadcasting, in that
    base broadcasting only supports `broadcast=true`.
"""
macro (..)(ex)
    fb_macro(ex, __module__, False(), False())
end

function __process_kwarg(kwarg)
    threadarg = kwarg.args[2]
    threadarg isa Bool ? (threadarg ? True() : False()) : threadarg
end
function _validate_kwarg(kwarg)
    @assert Meta.isexpr(kwarg, :(=), 2)
    argname = kwarg.args[1]
    @assert (argname === :thread) || (argname === :broadcast)
    argname === :thread
end
function _process_kwarg(kwarg, threadarg = False(), broadcastarg = False())
    if _validate_kwarg(kwarg)
        return __process_kwarg(kwarg), broadcastarg
    else
        return threadarg, __process_kwarg(kwarg)
    end
end

macro (..)(kwarg, ex)
    threadarg, broadcastarg = _process_kwarg(kwarg)
    fb_macro(ex, __module__, threadarg, broadcastarg)
end
macro (..)(kwarg0, kwarg1, ex)
    threadarg, broadcastarg = _process_kwarg(kwarg0)
    threadarg, broadcastarg = _process_kwarg(kwarg1, threadarg, broadcastarg)
    fb_macro(ex, __module__, threadarg, broadcastarg)
end

const DEBUG = Ref(false)
@generated function _fast_materialize!(
    dst,
    ::Val{NOALIAS},
    ::DB,
    bc::Broadcasted,
    dstaxes::Tuple{Vararg{Any,N}},
    ax,
    indexstyle,
) where {N,DB,NOALIAS}
    loopbody_lin = :($setindex!(dst))
    loopbody_car = :($setindex!(dst))
    loopbody_slow = :($setindex!(dst))
    bcc = BroadcastCharacteristics()
    ii = map(Base.Fix1(Symbol, :i_), 1:N)
    walk_bc!(bcc, loopbody_lin, loopbody_car, loopbody_slow, ii, bc, :bc, ax, :ax)
    push!(loopbody_lin.args, :i)
    append!(loopbody_car.args, ii)
    append!(loopbody_slow.args, ii)
    loop_quote = if N > 1 && !(bcc.maybelinear && (indexstyle === IndexLinear))
        loop = if NOALIAS
            quote
                @simd ivdep for i_1 in $static_axes(dst, 1)
                    $loopbody_car
                end
            end
        else
            quote
                @simd for i_1 in $static_axes(dst, 1)
                    $loopbody_car
                end
            end
        end
        for n = N:-1:2
            loop = quote
                for $(ii[n]) in $static_axes(dst, $n)
                    $loop
                end
            end
        end
        loop
    elseif NOALIAS
        quote
            @simd ivdep for i in $eachindex(dst)
                $loopbody_lin
            end
        end
    else
        quote
            @simd for i in $eachindex(dst)
                $loopbody_lin
            end
        end
    end
    q = quote
        $(Expr(:meta, :inline))
        isfast = true
        (Base.Cartesian.@ntuple $N dstaxis) = dstaxes
        $(bcc.loopheader)
    end
    if DB === False
        if DEBUG[]
            push!(
                q.args,
                quote
                    if isfast
                        @inbounds $loop_quote
                    else
                        throw(
                            ArgumentError(
                                "Could not avoid broadcasting, because an axis had runtime size=1. Axes were: $(ax).",
                            ),
                        )
                    end
                end,
            )
        else
            push!(q.args, :(@inbounds $loop_quote))
        end
    else
        push!(q.args, quote
            if isfast
                @inbounds $loop_quote
            else
                Base.Cartesian.@nloops $N i dst begin
                    $loopbody_slow
                end
            end
        end)
    end
    push!(q.args, :dst)
    return q
end
end
