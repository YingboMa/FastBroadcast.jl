module FastBroadcast

using Core: CodeInfo
using Base: UV_UNKNOWN
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
@inline _fastindex(A::AbstractArray, i::Int) = @inbounds A[i]
@inline function _fastindex(A::AbstractArray{<:Any,M}, i::CartesianIndex{N}) where {M,N}
  inds = _rmap(static_axes(A), to_tup(Val(M), i)) do ax, j
    Bool(_static_one(static_length(ax))) ? 1 : j
  end
  @inbounds A[inds...]
end


@inline _slowindex(b::Number, _) = b
@inline _slowindex(b::Tuple{X}, _) where {X} = only(b)
@inline _slowindex(b::Base.RefValue, _) = b[]
@inline _slowindex(b::Tuple{X,Y,Vararg}, i) where {X,Y} = @inbounds b[i]
@inline _slowindex(b::Broadcasted, i) = b.f(_rmap(Fix{2}(_slowindex, i), b.args)...)
@inline _slowindex(A::AbstractArray, i::Int) = @inbounds A[i]
@inline function _slowindex(A::AbstractArray{<:Any,M}, i::CartesianIndex{N}) where {M,N}
  inds = to_tup(Val(M), i)
  @inbounds A[_rmap(ifelse, _rmap(isone, size(A)), _rmap(first, axes(A)), inds)...]
end

@inline _all(_, x::Tuple{}) = True()
@inline _all(f, x::Tuple{X}) where {X} = f(only(x))
@inline __all(f::F, ::False, x::Tuple) where {F} = False()
@inline __all(f::F, ::True, x::Tuple) where {F} = _all(f, x)
@inline _all(f, x::Tuple{X,Y,Vararg}) where {X,Y} = __all(f, f(first(x)), Base.tail(x))
@inline _none_one(x) = _all(Fix{2}(Static.ne, Static.One()) ∘ static_length, x)

@inline _any(_, x::Tuple{}) = False()
@inline _any(f, x::Tuple{X}) where {X} = f(only(x))
@inline __any(f::F, ::True, x::Tuple) where {F} = True()
@inline __any(f::F, ::False, x::Tuple) where {F} = _any(f, x)
@inline _any(f, x::Tuple{X,Y,Vararg}) where {X,Y} = __any(f, f(first(x)), Base.tail(x))

@inline _static_one(::Static.One) = True()
@inline _static_one(::Any) = False()
@inline _any_one(x) = _any(_static_one ∘ static_length, x)

@inline _rall(_, x::Tuple{}) = true
@inline _rall(f, x::Tuple{X}) where {X} = f(only(x))
@inline _rall(f, x::Tuple{X,Y,Vararg}) where {X,Y} = f(first(x)) && _rall(f, Base.tail(x))
@inline _rall(x::Tuple) = _rall(identity, x)

@inline _rmap(_, ::Tuple{}) = ()
@inline _rmap(f, x::Tuple{X}) where {X} = (f(only(x)),)
@inline _rmap(f, x::Tuple{X,Y,Vararg}) where {X,Y} = (f(first(x)), _rmap(f, Base.tail(x))...)

@inline _rmap(_, ::Tuple{}, ::Tuple{}) = ()
@inline _rmap(f, a::Tuple{A}, x::Tuple{X}) where {A,X} = (@inline(f(only(a), only(x))),)
@inline _rmap(f, a::Tuple{A,B,Vararg}, x::Tuple{X,Y,Vararg}) where {A,B,X,Y} = (@inline(f(first(a), first(x))), _rmap(f, Base.tail(a), Base.tail(x))...)

@inline _rmap(_, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline _rmap(f, a::Tuple{A}, m::Tuple{M}, x::Tuple{X}) where {A,M,X} = (@inline(f(only(a), only(m), only(x))),)
@inline _rmap(f, a::Tuple{A,B,Vararg}, m::Tuple{M,N,Vararg}, x::Tuple{X,Y,Vararg}) where {A,B,M,N,X,Y} = (@inline(f(first(a), first(m), first(x))), _rmap(f, Base.tail(a), Base.tail(m), Base.tail(x))...)



@inline function _static_match(ax0::Tuple{Vararg{Any,M}}, ax1::Tuple{Vararg{Any,N}}) where {M,N}
  subax1 = ntuple(Fix{1}(Base.getindex, ax1), Val(M))
  eqs = _rmap(ax0, subax1) do x0, x1
    Bool(_static_one(static_length(x0))) || x0 == x1
  end
  _rall(eqs)
end
@inline function _dynamic_match(ax0::Tuple{Vararg{Any,M}}, ax1::Tuple{Vararg{Any,N}}) where {M,N}
  subax1 = ntuple(Fix{1}(Base.getindex, ax1), Val(M))
  eqs = _rmap(ax0, subax1) do x0, x1
    isone(length(x0)) || x0 == x1
  end
  _rall(eqs)
end

@inline function _static_not_flat(B::AbstractArray, ::Val{N}) where {N}
  (IndexStyle(typeof(B)) === IndexLinear()) && ndims(B) == N ? _any_one(static_axes(B)) : True()
end
@inline function _static_not_flat(bc::Broadcasted, VN::Val)
  _any(Fix{2}(_static_not_flat, VN), bc.args)
end

@inline _checkaxes(::Union{Number,Base.RefValue}, _) = true
@inline function _checkaxes(::Tuple{Vararg{Any,M}}, ax::Tuple{Vararg{Any,N}}) where {M,N}
  M == 1 || M == length(first(ax))
end
@inline function _checkaxes(B::AbstractArray{<:Any,M}, ax::Tuple{Vararg{Any,N}}) where {M,N}
  _dynamic_match(static_axes(B), ax)
end
@inline function _checkaxes(bc::Broadcasted, ax)
  _rall(_rmap(Fix{2}(_checkaxes, ax), bc.args))
end

@inline _static_checkaxes(::Union{Number,Base.RefValue}, _) = true, False()
@inline function _static_checkaxes(::Tuple{Vararg{Any,M}}, ax::Tuple{Vararg{Any,N}}) where {M,N}
  snf = N == 1 ? False() : True()
  equal_axes = M == 1 || M == length(first(ax))
  equal_axes, snf
end
@inline function _static_checkaxes(B::AbstractArray{<:Any,M}, ax::Tuple{Vararg{Any,N}}) where {M,N}
  snf = _static_not_flat(B, Val(N))
  equal_axes = _static_match(static_axes(B), ax)
  equal_axes, snf
end
@inline function _static_checkaxes(bc::Broadcasted, ax)
  tups = _rmap(Fix{2}(_static_checkaxes, ax), bc.args)
  _rall(first, tups), _any(last, tups)
end

@inline function __fast_materialize!(dst, ::Val{true}, bc::Broadcasted, ::False)
  @simd ivdep for i = eachindex(dst)
    @inbounds dst[i] = _fastindex(bc, i)
  end
  return dst
end
@inline function __fast_materialize!(dst, ::Val{true}, bc::Broadcasted, ::True)
  @simd ivdep for i = CartesianIndices(dst)
    @inbounds dst[i] = _fastindex(bc, i)
  end
  return dst
end
@inline function __fast_materialize!(dst, ::Val{false}, bc::Broadcasted, ::False)
  for i = eachindex(dst)
    @inbounds dst[i] = _fastindex(bc, i)
  end
  return dst
end
@inline function __fast_materialize!(dst, ::Val{false}, bc::Broadcasted, ::True)
  for i = CartesianIndices(dst)
    @inbounds dst[i] = _fastindex(bc, i)
  end
  return dst
end

@inline function _fast_materialize!(dst::AbstractArray{<:Any,N}, ::Val{NOALIAS}, ::False, bc::Broadcasted) where {N,NOALIAS}
  _no_dyn_broadcast, _islinear = _static_checkaxes(bc, static_axes(dst))
  _no_dyn_broadcast || throw(ArgumentError("Some axes are not equal, or feature a dynamic broadcast!"))
  __fast_materialize!(dst, Val(NOALIAS), bc, _islinear)
  return dst
end

@inline function _fast_materialize!(
  dst,
  ::Val{NOALIAS},
  ::True,
  bc::Broadcasted,
) where {NOALIAS}
  _no_dyn_broadcast, _islinear = _static_checkaxes(bc, static_axes(dst))
  _no_dyn_broadcast && return __fast_materialize!(dst, Val(NOALIAS), bc, _islinear)
  _checkaxes(bc, static_axes(dst)) || throw(ArgumentError("Size mismatch."))
  _slow_materialize!(dst, Val(NOALIAS), bc)
end

function _slow_materialize!(
  dst,
  ::Val{true},
  bc::Broadcasted,
)
  @simd ivdep for i = CartesianIndices(dst)
    @inbounds dst[i] = _slowindex(bc, i)
  end
  return dst
end
function _slow_materialize!(
  dst,
  ::Val{false},
  bc::Broadcasted,
)
  for i = CartesianIndices(dst)
    @inbounds dst[i] = _slowindex(bc, i)
  end
  return dst
end

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
use_fast_broadcast(_) = false
use_fast_broadcast(::Type{<:Base.Broadcast.DefaultArrayStyle}) = true
use_fast_broadcast(::Type{<:Base.Broadcast.DefaultArrayStyle{0}}) = false


@inline function fast_materialize!(::False, ::DB, dst::A, bc::Broadcasted{S}) where {S,DB,A}
  if use_fast_broadcast(S)
    _fast_materialize!(
      dst,
      Val(indices_do_not_alias(A)),
      DB(),
      bc,
    )
  else
    Base.Broadcast.materialize!(dst, bc)
  end
end

@inline _view(A::AbstractArray{<:Any,N}, r, ::Val{N}) where {N} =
  view(A, ntuple(_ -> :, N - 1)..., r)
@inline _view(A::AbstractArray, r, ::Val) = A
@inline _view(x, r, ::Val) = x
@inline __view(t::Tuple{T}, r, ::Val{N}) where {T,N} = (_view(first(t), r, Val(N)),)
@inline __view(t::Tuple{T,Vararg}, r, ::Val{N}) where {T,N} =
  (_view(first(t), r, Val(N)), __view(Base.tail(t), r, Val(N))...)
@inline function _view(bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle{N},Nothing}, r, ::Val{N},) where {N}
  Base.Broadcast.Broadcasted(bc.f, __view(bc.args, r, Val(N)))
end
@inline _view(bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle}, _, ::Val{N},) where {N} = bc
@inline _view(t::Tuple{Vararg{AbstractRange,N}}, r, ::Val{N}) where {N} = (Base.front(t)..., r)

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

function gotoifnot_stmt(base::Symbol, cond, dest::Int)
  ex = Expr(:||)
  pushsymname!(ex, base, cond)
  push!(ex.args, goto(base, dest))
  return ex
end
function _push_static_bool!(ex::Expr, b)
  if !(b isa Bool)
    push!(ex.args, b)
  elseif b
    push!(ex.args, True())
  else
    push!(ex.args, False())
  end
end

function broadcast_stmt!(gotos::Vector{Int}, base::Symbol, i::Int,
  threadarg, broadcastarg, @nospecialize(code))
  if Meta.isexpr(code, :call)
    ex = Expr(:call)
    f = code.args[1]
    if f === GlobalRef(Base, :materialize)
      push!(ex.args, fast_materialize)
      _push_static_bool!(ex, threadarg)
      _push_static_bool!(ex, broadcastarg)
    elseif f === GlobalRef(Base, :materialize!)
      push!(ex.args, fast_materialize!)
      _push_static_bool!(ex, threadarg)
      _push_static_bool!(ex, broadcastarg)
    elseif f === GlobalRef(Base, :getindex)
      push!(ex.args, Base.Broadcast.dotview)
    else
      pushsymname!(ex, base, f)
    end
    for arg ∈ @view(code.args[2:end])
      pushsymname!(ex, base, arg)
    end
    return Expr(:(=), Symbol(base, '_', i), ex)
  elseif Meta.isexpr(code, :(=))
    ex = Expr(:(=), Symbol(base, 's', code.args[1].id))
    rhs = broadcast_stmt!(gotos, base, i, threadarg, broadcastarg, code.args[2])
    pushsymname!(ex, base, rhs)
    return ex
  elseif VERSION ≥ v"1.6" && code isa Core.GotoIfNot
    push!(gotos, code.dest)
    return gotoifnot_stmt(base, code.cond, code.dest)
  elseif VERSION < v"1.6" && Meta.isexpr(code, :gotoifnot)
    cond, dest = code.args
    push!(gotos, dest)
    return gotoifnot_stmt(base, cond, dest)
  elseif code isa Core.GotoNode
    push!(gotos, code.label)
    ex = goto(base, code.label)
    return ex
  elseif !(VERSION ≥ v"1.6" ? isa(code, Core.ReturnNode) : Meta.isexpr(code, :return))
    ex = Expr(:(=), Symbol(base, '_', i))
    pushsymname!(ex, base, code)
    return ex
  end
  return nothing
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
    ex = broadcast_stmt!(gotos, base, i, threadarg, broadcastarg, code)
    if ex !== nothing
      push!(q.args, ex)
    end
  end
  q
end

function fb_macro(ex::Expr, mod::Module, threadarg, broadcastarg)
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
  ex isa Expr || return esc(ex)
  fb_macro(ex, __module__, false, false)
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
  fb_macro(ex, __module__, threadarg, broadcastarg)
end
macro (..)(kwarg0, kwarg1, ex)
  threadarg, broadcastarg = _process_kwarg(kwarg0)
  threadarg, broadcastarg = _process_kwarg(kwarg1, threadarg, broadcastarg)
  fb_macro(ex, __module__, threadarg, broadcastarg)
end

let # we could check `hasfield(Method, :recursion_relation)`, but I'd rather see an error if things change
  dont_limit = Returns(true)
  for f in (_fastindex, _slowindex, _static_not_flat, _checkaxes, _static_checkaxes, _fast_materialize!, __any, _any, __all, _all, _rall, _rmap, __view, _view)
    for m in methods(f)
      m.recursion_relation = dont_limit
    end
  end
end


end
