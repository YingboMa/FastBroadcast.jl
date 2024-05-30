module FastBroadcast

using Core: CodeInfo
using Base: UV_UNKNOWN, @propagate_inbounds, threadcall_restrictor
export @..

using StaticArrayInterface: static_axes, static_length
using ArrayInterface: indices_do_not_alias
using Base.Broadcast: Broadcasted
using LinearAlgebra: Adjoint, Transpose
using Static, Polyester
using StrideArraysCore: AbstractStrideArray

struct Fix{N,F,T}
  f::F
  x::T
  @inline Fix{N}(f::F, x::T) where {N,F,T} = new{N,F,T}(f, x)
end
@inline (i::Fix{1})(arg) = @inline i.f(i.x, arg)
@inline (i::Fix{2})(arg) = @inline i.f(arg, i.x)

@inline function to_tup(::Val{M}, i::CartesianIndex{N}) where {M,N}
  if M < N
    ntuple(Fix{1}(getindex, Tuple(i)), Val(M))
  else
    M == N || error("Array of higher dimension than cartesian index.")
    Tuple(i)
  end
end

@inline _fastindex(b::Number, _) = b
@inline _fastindex(b::Tuple{X}, _) where {X} = only(b)
@inline _fastindex(b::Base.RefValue, _) = b[]
@inline _fastindex(b::Tuple{X,Y,Vararg}, i) where {X,Y} = @inbounds b[i]
@inline _fastindex(b::Broadcasted, i) = b.f(_rmap(Fix{2}(_fastindex, i), b.args)...)
@inline function _fastindex(A, i)
  i isa Int && return @inbounds A[i]
  axs = static_axes(A)
  inds = _rmap(axs, to_tup(Val(length(axs)), i)) do ax, j
    Bool(_static_one(static_length(ax))) ? 1 : j
  end
  @inbounds A[inds...]
end

@inline _slowindex(b::Number, _) = b
@inline _slowindex(b::Tuple{X}, _) where {X} = only(b)
@inline _slowindex(b::Base.RefValue, _) = b[]
@inline _slowindex(b::Tuple{X,Y,Vararg}, i) where {X,Y} = @inbounds b[i]
@inline _slowindex(b::Broadcasted, i) = b.f(_rmap(Fix{2}(_slowindex, i), b.args)...)
@inline function _slowindex(A, i)
  i isa Int && return @inbounds A[i]
  axs = static_axes(A)
  @inbounds A[_rmap(
    ifelse, _rmap(isone, size(A)), _rmap(first, axs), to_tup(Val(length(axs)), i))...]
end

@inline _all(@nospecialize(_), x::Tuple{}) = True()
@inline _all(f, x::Tuple{X}) where {X} = f(only(x))
@inline __all(@nospecialize(_), ::False, x::Tuple) = False()
@inline __all(f::F, ::True, x::Tuple) where {F} = _all(f, x)
@inline _all(f, x::Tuple{X,Y,Vararg}) where {X,Y} = __all(f, f(first(x)), Base.tail(x))
@inline _none_one(x) = _all(Fix{2}(Static.ne, Static.One()) ∘ static_length, x)

@inline _any(@nospecialize(_), x::Tuple{}) = False()
@inline _any(f, x::Tuple{X}) where {X} = f(only(x))
@inline __any(@nospecialize(_), ::True, x::Tuple) = True()
@inline __any(f::F, ::False, x::Tuple) where {F} = _any(f, x)
@inline _any(f, x::Tuple{X,Y,Vararg}) where {X,Y} = __any(f, f(first(x)), Base.tail(x))

@inline _static_one(::Static.One) = True()
@inline _static_one(::Any) = False()
@inline _any_one(x) = _any(_static_one ∘ static_length, x)

@inline _rall(@nospecialize(_), x::Tuple{}) = true
@inline _rall(f, x::Tuple{X}) where {X} = f(only(x))
@inline _rall(f, x::Tuple{X,Y,Vararg}) where {X,Y} = f(first(x)) &&
                                                     _rall(f, Base.tail(x))
@inline _rall(x::Tuple) = _rall(identity, x)

@inline _rmap(@nospecialize(_), ::Tuple{}) = ()
@inline _rmap(f, x::Tuple{X}) where {X} = (f(only(x)),)
@inline _rmap(f, x::Tuple{X,Y,Vararg}) where {X,Y} = (
  f(first(x)), _rmap(f, Base.tail(x))...)

@inline _rmap(@nospecialize(_), ::Tuple{}, ::Tuple{}) = ()
@inline _rmap(f, a::Tuple{A}, x::Tuple{X}) where {A,X} = (@inline(f(only(a), only(x))),)
@inline _rmap(f, a::Tuple{A,B,Vararg}, x::Tuple{X,Y,Vararg}) where {A,B,X,Y} = (
  @inline(f(first(a), first(x))), _rmap(f, Base.tail(a), Base.tail(x))...)

@inline _rmap(@nospecialize(_), ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline _rmap(f, a::Tuple{A}, m::Tuple{M}, x::Tuple{X}) where {A,M,X} = (@inline(f(
  only(a), only(m), only(x))),)
@inline _rmap(f, a::Tuple{A,B,Vararg}, m::Tuple{M,N,Vararg}, x::Tuple{X,Y,Vararg}) where {A,B,M,N,X,Y} = (
  @inline(f(first(a), first(m), first(x))),
  _rmap(f, Base.tail(a), Base.tail(m), Base.tail(x))...)

@inline function _static_match(
  ax0::Tuple{Vararg{Any,M}}, ax1::Tuple{Vararg{Any,N}}) where {M,N}
  subax1 = ntuple(Fix{1}(Base.getindex, ax1), Val(M))
  eqs = _rmap(ax0, subax1) do x0, x1
    Bool(_static_one(static_length(x0))) || x0 == x1
  end
  _rall(eqs)
end
@inline function _dynamic_match(
  ax0::Tuple{Vararg{Any,M}}, ax1::Tuple{Vararg{Any,N}}) where {M,N}
  subax1 = ntuple(Fix{1}(Base.getindex, ax1), Val(M))
  eqs = _rmap(ax0, subax1) do x0, x1
    isone(length(x0)) || x0 == x1
  end
  _rall(eqs)
end

@inline _checkaxes(::Union{Number,Base.RefValue}, _) = true
@inline function _checkaxes(::Tuple{Vararg{Any,M}}, ax) where {M}
  M == 1 || M == length(first(ax))
end
@inline function _checkaxes(B, ax)
  _dynamic_match(static_axes(B), ax)
end
@inline function _checkaxes(bc::Broadcasted, ax)
  _rall(_rmap(Fix{2}(_checkaxes, ax), bc.args))
end

@inline _static_checkaxes(::Union{Number,Base.RefValue}, ::Tuple{Vararg{Any,N}}) where {N} = true,
False()
@inline function _static_checkaxes(
  ::Tuple{Vararg{Any,M}}, ax::Tuple{Vararg{Any,N}}) where {M,N}
  M == 1 || M == length(first(ax)), N == 1 ? False() : True()
end
@inline function _static_checkaxes(B, ax::Tuple{Vararg{Any,N}}) where {N}
  bx = static_axes(B)
  _static_match(bx, ax),
  (IndexStyle(typeof(B)) === IndexLinear()) && length(bx) == N ? _any_one(bx) : True()
end
@inline function _static_checkaxes(bc::Broadcasted, ax::Tuple{Vararg{Any,N}}) where {N}
  tups = _rmap(Fix{2}(_static_checkaxes, ax), bc.args)
  _rall(first, tups), _any(last, tups)
end

@inline function __fast_materialize!(dst, ::Val{true}, bc::Broadcasted, ::False)
  @simd ivdep for i in eachindex(dst)
    @inbounds dst[i] = _fastindex(bc, i)
  end
  return dst
end
@inline function __fast_materialize!(dst, ::Val{true}, bc::Broadcasted, ::True)
  @simd ivdep for i in CartesianIndices(dst)
    @inbounds dst[i] = _fastindex(bc, i)
  end
  return dst
end
@inline function __fast_materialize!(dst, ::Val{false}, bc::Broadcasted, ::False)
  for i in eachindex(dst)
    @inbounds dst[i] = _fastindex(bc, i)
  end
  return dst
end
@inline function __fast_materialize!(dst, ::Val{false}, bc::Broadcasted, ::True)
  for i in CartesianIndices(dst)
    @inbounds dst[i] = _fastindex(bc, i)
  end
  return dst
end

@inline function _fast_materialize!(
  dst, ::Val{NOALIAS}, ::False, bc::Broadcasted) where {NOALIAS}
  _no_dyn_broadcast, _islinear = _static_checkaxes(bc, static_axes(dst))
  @boundscheck _no_dyn_broadcast ||
               throw(ArgumentError("Some axes are not equal, or feature a dynamic broadcast!"))
  __fast_materialize!(dst, Val(NOALIAS), bc, _islinear)
  return dst
end

@inline function _fast_materialize!(
  dst, ::Val{NOALIAS}, ::True, bc::Broadcasted) where {NOALIAS}
  sad = static_axes(dst)
  _no_dyn_broadcast, _islinear = _static_checkaxes(bc, sad)
  _no_dyn_broadcast && return __fast_materialize!(dst, Val(NOALIAS), bc, _islinear)
  @boundscheck _checkaxes(bc, sad) || throw(ArgumentError("Size mismatch."))
  _slow_materialize!(dst, Val(NOALIAS), bc)
end
fast_materialize!(_, _, dst, x::Number) = fill!(dst, x)
fast_materialize!(_, ::False, dst, x::AbstractArray) = copyto!(dst, x)
function fast_materialize!(_, ::True, dst, x::AbstractArray)
  sad = static_axes(dst)
  _no_dyn_broadcast, _islinear = _static_checkaxes(bc, sad)
  _no_dyn_broadcast && return copyto!(dst, x)
  @boundscheck _checkaxes(bc, sad) || throw(ArgumentError("Size mismatch."))
  for i in CartesianIndices(dst)
    @inbounds dst[i] = _slowindex(x, i)
  end
  return dst
end

function _slow_materialize!(
  dst,
  ::Val{true},
  bc::Broadcasted
)
  @simd ivdep for i in CartesianIndices(dst)
    @inbounds dst[i] = _slowindex(bc, i)
  end
  return dst
end
function _slow_materialize!(
  dst,
  ::Val{false},
  bc::Broadcasted
)
  for i in CartesianIndices(dst)
    @inbounds dst[i] = _slowindex(bc, i)
  end
  return dst
end

Base.@propagate_inbounds function fast_materialize(
  ::SB, ::DB, bc::Broadcasted{S}) where {S,SB,DB}
  if use_fast_broadcast(S)
    fast_materialize!(
      SB(), DB(), similar(bc, Base.Broadcast.combine_eltypes(bc.f, bc.args)), bc)
  else
    Base.Broadcast.materialize(bc)
  end
end
use_fast_broadcast(_) = false
use_fast_broadcast(::Type{<:Base.Broadcast.DefaultArrayStyle}) = true
use_fast_broadcast(::Type{<:Base.Broadcast.DefaultArrayStyle{0}}) = false

Base.@propagate_inbounds function fast_materialize!(
  ::False, ::DB, dst::A, bc::Broadcasted{S}) where {S,DB,A}
  if use_fast_broadcast(S)
    _fast_materialize!(dst, Val(indices_do_not_alias(A)), DB(), bc)
  else
    Base.Broadcast.materialize!(dst, bc)
  end
end

@inline _view(A::AbstractArray{<:Any,N}, r, ::Val{N}) where {N} = view(
  A, ntuple(_ -> :, N - 1)..., r)
@inline _view(A::AbstractArray, r, ::Val) = A
@inline _view(x, r, ::Val) = x
@inline __view(t::Tuple{T}, r, ::Val{N}) where {T,N} = (_view(first(t), r, Val(N)),)
@inline __view(t::Tuple{T,Vararg}, r, ::Val{N}) where {T,N} = (
  _view(first(t), r, Val(N)), __view(Base.tail(t), r, Val(N))...)
@inline function _view(
  bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle{N},Nothing},
  r, ::Val{N}) where {N}
  Base.Broadcast.Broadcasted(bc.f, __view(bc.args, r, Val(N)))
end
@inline _view(bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle}, _, ::Val{N}) where {N} = bc
@inline _view(t::Tuple{Vararg{AbstractRange,N}}, r, ::Val{N}) where {N} = (
  Base.front(t)..., r)

@inline function fast_materialize!(::True, ::DB, dst, bc::Broadcasted{S}) where {S,DB}
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
@inline function fast_materialize_threaded!(
  dst,
  ::DB,
  bc::Broadcasted,
  dstaxes::Tuple{Vararg{Any,N}}
) where {N,DB}
  last_dstaxes = dstaxes[N]
  Polyester.batch(_batch_broadcast_fn, (length(last_dstaxes), Threads.nthreads()),
    dst, last_dstaxes, bc, Val(N), DB())
  return dst
end

function _pushfirst_static!(x, b)
  if !(b isa Bool)
    pushfirst!(x, b)
  elseif b
    pushfirst!(x, True())
  else
    pushfirst!(x, False())
  end
end

@inline _broadcasted(f::F, args...) where {F} = Base.Broadcast.broadcasted(f, args...)
@inline _broadcasted(::Colon, arg0, arg1) = arg0:arg1
@inline _broadcasted(::typeof(Base.maybeview), args...) = begin
  Base.maybeview(args...)
end

function _fb_macro!(ex::Expr, threadarg, broadcastarg)
  ops = (:(+), :(-), :(*), :(/), :(\), :(÷), :(&),
    :(|), :(⊻), :(>>), :(>>>), :(<<))
  if Meta.isexpr(ex, :(.)) && length(ex.args) == 2
    args = ex.args[2]
    args isa QuoteNode && return
    resize!(ex.args, 1)
    ex.head = :call
    append!(ex.args, args.args)
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
    ind = Base.findfirst(
      ==(ex.args[2]), (Symbol(".+"), Symbol(".-"), Symbol(".*"),
        Symbol("./"), Symbol(".\\"), Symbol(".÷"), Symbol(".&"),
        Symbol(".|"), Symbol(".⊻"), Symbol(".>>"), Symbol(".>>>"), Symbol(".<<")))
    if ind !== nothing
      ex.args[2] = ops[ind]
    end
  elseif Meta.isexpr(ex, :(=))
    ex.head = :call
    _pushfirst_static!(ex.args, broadcastarg)
    _pushfirst_static!(ex.args, threadarg)
    pushfirst!(ex.args, fast_materialize!)
    a4 = ex.args[4]
    if Meta.isexpr(a4, :ref)
      a4.head = :call
      pushfirst!(a4.args, Base.maybeview)
      skip = 4
    end
  elseif Meta.isexpr(ex, :($), 1)
    if (exarg = only(ex.args); exarg isa Expr)
      ex.head = exarg.head
      ex.args = exarg.args
      return
    end
  elseif Meta.isexpr(ex, :let)
    skip = 1
  else
    ind = Base.findfirst(
      ==(ex.head), (:(+=), :(-=), :(*=), :(/=), :(\=), :(÷=), :(&=),
        :(|=), :(⊻=), :(>>=), :(>>>=), :(<<=)))
    if ind !== nothing
      op = ops[ind]
      # x op= f(args...) -> x = op(x, f(args...))
      @assert length(ex.args) == 2
      ex.args[2] = Expr(:call, op, ex.args[1], ex.args[2])
      ex.head = :(=)
      return _fb_macro!(ex, threadarg, broadcastarg)
    end
  end
  for i in (1+skip):length(ex.args)
    x = ex.args[i]
    x isa Expr &&
      _fb_macro!(x, threadarg, broadcastarg)
  end
end
function fb_macro!(ex::Expr, threadarg, broadcastarg)
  iscall = Meta.isexpr(ex, :call)
  _fb_macro!(ex, threadarg, broadcastarg)
  if iscall
    ex = Expr(:call, ex)
    _pushfirst_static!(ex.args, broadcastarg)
    _pushfirst_static!(ex.args, threadarg)
    pushfirst!(ex.args, fast_materialize)
  end
  esc(ex)
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
function _process_kwarg(kwarg, threadarg=false, broadcastarg=false)
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
  for f in (_fastindex, _slowindex, _checkaxes, _static_checkaxes,
    __any, _any, __all, _all, _rall, _rmap, __view, _view)
    for m in methods(f)
      m.recursion_relation = dont_limit
    end
  end
end

end
