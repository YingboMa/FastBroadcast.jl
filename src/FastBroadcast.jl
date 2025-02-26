module FastBroadcast

using Base: @propagate_inbounds, Fix1, Fix2
using Base.Broadcast: Broadcasted, materialize, materialize!

using ArrayInterface: indices_do_not_alias, flatten_tuples
using StaticArrayInterface: static_axes, static_length
using LinearAlgebra: Adjoint, Transpose
using Polyester

export @..

@inline function to_tup(::Val{M}, i::CartesianIndex{N}) where {M, N}
    if M < N
        ntuple(Fix1(getindex, Tuple(i)), Val(M))
    elseif M == N
        Tuple(i)
    else # M > N
        (Tuple(i)..., ntuple(one, Val(M - N))...)
    end
end

@inline _rall(x::Tuple{}) = true
@inline _rall(x::Tuple{Vararg{Bool}}) = first(x) && _rall(Base.tail(x))

@inline _rany(x::Tuple{}) = false
@inline _rany(x::Tuple{Vararg{Bool}}) = (first(x)) || _rany(Base.tail(x))

@inline _rmap(f, ::Tuple{}) = ()
@inline _rmap(f, x::Tuple{X}) where {X} = (@inline(f(only(x))),)
@inline _rmap(f, x::Tuple{X, Y, Vararg}) where {X, Y} = (
    @inline(f(first(x))), _rmap(f, Base.tail(x))...)

@inline _rmap(f, ::Tuple{}, ::Tuple{}) = ()
@inline _rmap(f, a::Tuple{A}, x::Tuple{X}) where {A, X} = (@inline(f(only(a), only(x))),)
@inline _rmap(f, a::Tuple{A, B, Vararg}, x::Tuple{X, Y, Vararg}) where {A, B, X, Y} = (
    @inline(f(first(a), first(x))), _rmap(f, Base.tail(a), Base.tail(x))...)
@inline _rmap(
    f,
    a::Tuple{A, B, Vararg},
    m::Tuple{M, N, Vararg},
    x::Tuple{X, Y, Vararg}
    ) where {A, B, M, N, X, Y} = (
    @inline(f(first(a), first(m), first(x))),
    _rmap(f, Base.tail(a), Base.tail(m), Base.tail(x))...
)

struct SingularAxis
end
function axismatch(ax1, ax2)
    ax1 isa SingularAxis && return true
    ax2 isa SingularAxis && return true
    length(ax1) == length(ax2)
end

const VecAdjTrans = Union{Transpose{T, M}, Adjoint{T, M}} where {T, M<:AbstractVector}
semi_static_axes(A) = Base.axes(A)
semi_static_axes(A::VecAdjTrans)  = (SingularAxis(), only(semi_static_axes(parent(A))))
semi_static_axes(A::Tuple{T}) where T = (SingularAxis(),)
function semi_static_axes(S::SubArray)
    parent_axes = semi_static_axes(parent(S))
    flatten_tuples(_rmap(singularize_singular_slice, parent_axes, S.indices))
end
function singularize_singular_slice(parent_ax, ax)
    ax isa Integer ? () : ax isa Base.Slice ? parent_ax : ax
end

check_no_dyn_broadcast(::Union{Number, Base.RefValue}, ::Tuple{Vararg{Any, N}}) where {N} = true
@inline function check_no_dyn_broadcast(B, ax::Tuple{Vararg{Any, N}}) where {N}
    bx = semi_static_axes(B)
    MN = min(length(ax), length(bx))
    subax = ntuple(Fix1(Base.getindex, ax), Val(MN))
    subbx = ntuple(Fix1(Base.getindex, bx), Val(MN))
    _rall(_rmap(axismatch, subax, subbx))
end

@inline function check_no_dyn_broadcast(bc::Broadcasted, ax::Tuple{Vararg{Any, N}}) where {N}
    _rall(_rmap(check_no_dyn_broadcast, bc.args, ntuple(Returns(ax), Val(length(bc.args)))))
end

broadcast_islinear(::Union{Number, Base.RefValue}, ::Tuple{Vararg{Any, N}}) where {N} = false
function broadcast_islinear(::Tuple{Vararg{Any, M}}, ::Tuple{Vararg{Any, N}}) where {M, N}
    M != 1
end

function broadcast_islinear(B, ax::Tuple{Vararg{Any, N}}) where {N}
    bx = semi_static_axes(B)
    if (IndexStyle(typeof(B)) === IndexLinear()) && length(bx) == N
        _rany(_rmap(isequal(SingularAxis()), bx))
    else
        true
    end
end

function broadcast_islinear(bc::Broadcasted, ax::Tuple{Vararg{Any, N}}) where N
    _rany(_rmap(broadcast_islinear, bc.args, ntuple(Returns(ax), Val(length(bc.args)))))
end

@inline _fastindex(b::Number, _) = b
@inline _fastindex(b::Tuple{X}, _) where {X} = only(b)
@inline _fastindex(b::Base.RefValue, _) = b[]
@inline _fastindex(b::Tuple{X, Y, Vararg}, i) where {X, Y} = @inbounds b[i]
@inline _fastindex(b::Broadcasted, i) = b.f(_rmap(Fix2(_fastindex, i), b.args)...)
@inline function _fastindex(A, i)
    i isa Int && return @inbounds A[i]
    axs = semi_static_axes(A)
    inds = _rmap(axs, to_tup(Val(length(axs)), i)) do ax, j
        ax isa SingularAxis ? 1 : j
    end
    @inbounds A[inds...]
end

@inline function __fast_materialize!(dst, ::Val{true}, bc::Broadcasted, ::Val{false})
    @simd ivdep for i in eachindex(dst)
        @inbounds dst[i] = _fastindex(bc, i)
    end
    return dst
end
@inline function __fast_materialize!(dst, ::Val{true}, bc::Broadcasted, ::Val{true})
    @simd ivdep for i in CartesianIndices(dst)
        @inbounds dst[i] = _fastindex(bc, i)
    end
    return dst
end
@inline function __fast_materialize!(dst, ::Val{false}, bc::Broadcasted, ::Val{false})
    for i in eachindex(dst)
        @inbounds dst[i] = _fastindex(bc, i)
    end
    return dst
end
@inline function __fast_materialize!(dst, ::Val{false}, bc::Broadcasted, ::Val{true})
    for i in CartesianIndices(dst)
        @inbounds dst[i] = _fastindex(bc, i)
    end
    return dst
end

@inline function _fast_materialize!(dst, ::Val{NOALIAS},  bc::Broadcasted) where {NOALIAS}
    dstax = semi_static_axes(dst)
    _no_dyn_broadcast = check_no_dyn_broadcast(bc, semi_static_axes(dst))
    @boundscheck _no_dyn_broadcast || throw(
        DimensionMismatch("Some axes are not equal, or feature a dynamic broadcast!"),
    )
    _islinear = broadcast_islinear(bc, dstax)
    __fast_materialize!(dst, Val(NOALIAS), bc, Val(_islinear))
    return dst
end

# these are needed for `RecursiveArrayTools`'s fast broadcast extension
use_fast_broadcast(_) = false
use_fast_broadcast(::Type{<:Base.Broadcast.DefaultArrayStyle}) = true
use_fast_broadcast(::Type{<:Base.Broadcast.DefaultArrayStyle{0}}) = false

fast_materialize!(dst, x::Number) = fill!(dst, x)
function fast_materialize!(dst, x::AbstractArray)
    Base.Broadcast.check_broadcast_shape(size(dst), size(x))
    copyto!(dst, x)
end

fast_materialize!(_, _, dst, x) = dst .= x

Base.@propagate_inbounds function fast_materialize(
        bc::Broadcasted{S}) where {S}
    if S === Base.Broadcast.DefaultArrayStyle{0}
        return only(bc)
    elseif S <: Base.Broadcast.DefaultArrayStyle
        fast_materialize!(
            similar(bc, Base.Broadcast.combine_eltypes(bc.f, bc.args)),
            bc
        )
    else
        materialize(bc)
    end
end

fast_materialize(@nospecialize(x)) = x

Base.@propagate_inbounds function fast_materialize!(
        dst::A, bc::Broadcasted{S}) where {S, A}
    if S === Base.Broadcast.DefaultArrayStyle{0}
        fill!(dst, bc[1])
    elseif S <: Base.Broadcast.DefaultArrayStyle
        _fast_materialize!(dst, Val(indices_do_not_alias(A)), bc)
    else
        materialize!(dst, bc)
    end
end

@inline _view(A::AbstractArray{<:Any, N}, r, ::Val{N}) where {N} = view(
    A, ntuple(_ -> :, N - 1)..., r)
@inline _view(A::AbstractArray, r, ::Val) = A
@inline _view(x, r, ::Val) = x
@inline __view(t::Tuple{T}, r, ::Val{N}) where {T, N} = (_view(first(t), r, Val(N)),)
@inline __view(t::Tuple{T, Vararg}, r, ::Val{N}) where {T, N} = (
    _view(first(t), r, Val(N)), __view(Base.tail(t), r, Val(N))...)
@inline function _view(
        bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle{N}, Nothing},
        r,
        ::Val{N}
) where {N}
    Base.Broadcast.Broadcasted(bc.f, __view(bc.args, r, Val(N)))
end
@inline _view(bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle}, _, ::Val{N}) where {N} = bc
@inline _view(t::Tuple{Vararg{AbstractRange, N}}, r, ::Val{N}) where {N} = (Base.front(t)..., r)

@inline function _batch_broadcast_fn(tup, start, stop)
    (dest, ldstaxes, bcobj, VN) = tup
    r = @inbounds ldstaxes[start:stop]
    fast_materialize!(_view(dest, r, VN), _view(bcobj, r, VN))
    return nothing
end
@inline function fast_materialize_threaded!(dst,bc::Broadcasted)
    dstaxes = axes(dst)
    last_dstaxes = dstaxes[end]
    Polyester.batch(
        _batch_broadcast_fn,
        (length(last_dstaxes), Threads.nthreads()),
        dst,
        last_dstaxes,
        bc,
        Val(length(dstaxes))
    )
    return dst
end

_dim0(_) = false
_dim0(::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}}) = true
@inline function _broadcasted(f::F, args...) where {F}
    bc = Base.Broadcast.broadcasted(f, args...)
    _dim0(bc) ? only(bc) : bc
end
@inline _broadcasted(::Colon, arg0, arg1) = arg0:arg1
@inline _broadcasted(::typeof(Base.maybeview), args...) = begin
    Base.maybeview(args...)
end

function _view!(ex::Expr)
    r = Expr(:ref)
    r.args = ex.args
    ex.head = :macrocall
    # `maybeview` doesn't return a `Broadcasted` object
    ex.args = Any[Symbol("@views"), Base.LineNumberNode(@__LINE__, @__FILE__), r]
    return nothing
end

@inline function literal_pow(x, ::Val{N}) where {N}
    if N == 0
        return one(x)
    elseif N == 1
        return x
    elseif N < 0
        return inv(literal_pow(x, Val(-N)))
    end
    p = N
    tz = trailing_zeros(p) + 1
    p >>= tz
    for _ in 1:(tz - 1)
        @fastmath x *= x
    end
    y = x
    @fastmath while p != 0
        tz = trailing_zeros(p) + 1
        p >>= tz
        for _ in 1:tz
            x *= x
        end
        y *= x
    end
    return y
end


function _fb_macro!(ex::Expr, threadarg, broadcastarg)
    ops = (:(+), :(-), :(*), :(/), :(\), :(÷), :(&), :(|), :(⊻), :(>>), :(>>>), :(<<), :(^))
    if Meta.isexpr(ex, :(.)) && length(ex.args) == 2
        args = ex.args[2]
        args isa QuoteNode && return
        resize!(ex.args, 1)
        ex.head = :call
        append!(ex.args, args.args)
    elseif Meta.isexpr(ex, :macrocall, 3) && ex.args[1] === Symbol("@view")
        ex3 = ex.args[3]
        ex.head = ex3.head
        ex.args = ex3.args
    end
    skip = 0
    if Meta.isexpr(ex, :call)
        if Meta.isexpr(ex.args[1], :($))
            ex.args[1] = only(ex.args[1].args)
            return
        elseif ex.args[1] == :(:)
            return
        end
        pushfirst!(ex.args, _broadcasted)
        fun = ex.args[2]
        ind = Base.findfirst(
            ==(fun),
            (
                Symbol(".+"),
                Symbol(".-"),
                Symbol(".*"),
                Symbol("./"),
                Symbol(".\\"),
                Symbol(".÷"),
                Symbol(".&"),
                Symbol(".|"),
                Symbol(".⊻"),
                Symbol(".>>"),
                Symbol(".>>>"),
                Symbol(".<<"),
                Symbol(".^")
            )
        )
        if ind !== nothing
            ex.args[2] = fun = ops[ind]
        end
        if fun === :(^) && length(ex.args) == 4 && (p = ex.args[4]; p isa Int)
            ex.args[2] = literal_pow
            ex.args[4] = Val(p)
        end
    elseif Meta.isexpr(ex, :(=))
        ex.head = :call
        if broadcastarg
            pushfirst!(ex.args, materialize!)
        elseif threadarg
            pushfirst!(ex.args, fast_materialize_threaded!)
        else
            pushfirst!(ex.args, fast_materialize!)
        end
        a4 = ex.args[end]
        if Meta.isexpr(a4, :ref)
            _view!(a4)
            skip = 4
        end
    elseif Meta.isexpr(ex, :($), 1)
        if (exarg = only(ex.args); exarg isa Expr)
            ex.head = exarg.head
            ex.args = exarg.args
            return
        end
    elseif Meta.isexpr(ex, :ref)
        return _view!(ex)
    elseif Meta.isexpr(ex, :let)
        skip = 1
    else
        ind = Base.findfirst(
            ==(ex.head),
            (
                :(+=),
                :(-=),
                :(*=),
                :(/=),
                :(\=),
                :(÷=),
                :(&=),
                :(|=),
                :(⊻=),
                :(>>=),
                :(>>>=),
                :(<<=),
                :(^=)
            )
        )
        if ind !== nothing
            op = ops[ind]
            # x op= f(args...) -> x = op(x, f(args...))
            @assert length(ex.args) == 2
            ex.args[2] = Expr(:call, op, ex.args[1], ex.args[2])
            ex.head = :(=)
            return _fb_macro!(ex, threadarg, broadcastarg)
        end
    end
    for i in (1 + skip):length(ex.args)
        x = ex.args[i]
        x isa Expr && _fb_macro!(x, threadarg, broadcastarg)
    end
end

function fb_macro!(ex::Expr, threadarg, broadcastarg)
    iscall = Meta.isexpr(ex, :call)
    _fb_macro!(ex, threadarg, broadcastarg)
    if iscall
        ex = Expr(:call, ex)
        if broadcastarg
            pushfirst!(ex.args, materialize)
        elseif threadarg
            pushfirst!(ex.args, fast_materialize_threaded)
        else
            pushfirst!(ex.args, fast_materialize)
        end
    end
    esc(ex)
end

"""
    @.. [thread=false] [broadcast=false] expr

`@..` turns `expr` into a broadcast-like expression, similar to `@.`.
It additionally provides two optional keyword arguments:

- thread: Defaults to `false`. Use multithreading? This only works if broadcast=false.
- broadcast: Defaults to `false`. If `true`, it will broadcast axes with dynamic
    runtime sizes of `1` to larger sizes, if `false` only sizes known to be `1`
    at compile time will be supported, i.e. axes such that
    `ArrayInterface.known_length(typeof(StaticArrayInterface.static_axes(x,i))) == 1` will
    be broadcast. Note that this differs from base broadcasting, in that
    base broadcasting only supports `broadcast=true`.
"""
macro (..)(ex)
    ex isa Expr || return esc(ex)
    fb_macro!(ex, false, false)
end

function __process_kwarg(kwarg)
    kwarg.args[2]
end
function _validate_kwarg(kwarg)
    @assert Meta.isexpr(kwarg, :(=), 2)
    argname = kwarg.args[1]
    @assert (argname === :thread) || (argname === :broadcast)
    argname === :thread
end
function _process_kwarg(kwarg, threadarg = false, broadcastarg = false)
    if _validate_kwarg(kwarg)
        return __process_kwarg(kwarg), broadcastarg
    else
        return threadarg, __process_kwarg(kwarg)
    end
end

macro (..)(kwarg, ex)
    threadarg, broadcastarg = _process_kwarg(kwarg)
    fb_macro!(ex, threadarg, broadcastarg)
end
macro (..)(kwarg0, kwarg1, ex)
    threadarg, broadcastarg = _process_kwarg(kwarg0)
    threadarg, broadcastarg = _process_kwarg(kwarg1, threadarg, broadcastarg)
    fb_macro!(ex, threadarg, broadcastarg)
end



let # we could check `hasfield(Method, :recursion_relation)`, but I'd rather see an error if things change
    dont_limit = Returns(true)
    for f in (
        check_no_dyn_broadcast,
        broadcast_islinear,
        semi_static_axes,
        __view,
        _view,
        _rall,
        _rany,
        _rmap,
        _fastindex,
    )
        for m in methods(f)
            m.recursion_relation = dont_limit
        end
    end
end

end
