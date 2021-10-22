# FastBroadcast

[![Build Status](https://github.com/YingboMa/FastBroadcast.jl/workflows/CI/badge.svg)](https://github.com/YingboMa/FastBroadcast.jl/actions)
[![Coverage](https://codecov.io/gh/YingboMa/FastBroadcast.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/YingboMa/FastBroadcast.jl)

FastBroadcast.jl exports `@..` that compiles broadcast expressions into loops
that are easier for the compiler to optimize.

```julia
julia> using FastBroadcast

julia> function fast_foo9(a, b, c, d, e, f, g, h, i)
           @.. a = b + 0.1 * (0.2c + 0.3d + 0.4e + 0.5f + 0.6g + 0.6h + 0.6i)
           nothing
       end
fast_foo9 (generic function with 1 method)

julia> function foo9(a, b, c, d, e, f, g, h, i)
           @. a = b + 0.1 * (0.2c + 0.3d + 0.4e + 0.5f + 0.6g + 0.6h + 0.6i)
           nothing
       end
foo9 (generic function with 1 method)

julia> a, b, c, d, e, f, g, h, i = [rand(100, 100, 2) for i in 1:9];

julia> using BenchmarkTools

julia> @btime fast_foo9($a, $b, $c, $d, $e, $f, $g, $h, $i);
  19.902 μs (0 allocations: 0 bytes)

julia> @btime foo9($a, $b, $c, $d, $e, $f, $g, $h, $i);
  81.457 μs (0 allocations: 0 bytes)
```

It's important to note that FastBroadcast doesn't speed up "dynamic broadcast",
i.e. when the arguments are not equal-axised or scalars. For example, dynamic
broadcast happens when the expansion of singleton dimensions occurs:

```julia
julia> b = [1.0];

julia> @btime foo9($a, $b, $c, $d, $e, $f, $g, $h, $i);
  70.634 μs (0 allocations: 0 bytes)

julia> @btime fast_foo9($a, $b, $c, $d, $e, $f, $g, $h, $i);
  131.470 μs (0 allocations: 0 bytes)
```

The macro `@..` of FastBroadcast.jl accepts a keyword argument `thread` 
determining whether the broadcast call should use threading (disabled
by default). You can use it as follows (starting Julia with multiple 
threads).
```julia
julia> using FastBroadcast

julia> function foo_serial!(dest, src)
           @.. thread=false dest = log(src)
       end
foo_serial! (generic function with 1 method)

julia> function foo_parallel!(dest, src)
           @.. thread=true dest = log(src)
       end
foo_parallel! (generic function with 1 method)

julia> function foo_maybe_parallel!(dest, src, thread)
           @.. thread=thread dest = log(src)
       end
foo_maybe_parallel! (generic function with 1 method)

julia> src = rand(10^4); dest = similar(src);

julia> @btime foo_serial!($dest, $src);
  50.860 μs (0 allocations: 0 bytes)

julia> @btime foo_parallel!($dest, $src);
  17.245 μs (1 allocation: 48 bytes)

julia> @btime foo_maybe_parallel!($dest, $src, $FastBroadcast.False());
  51.682 μs (0 allocations: 0 bytes)

julia> @btime foo_maybe_parallel!($dest, $src, $FastBroadcast.True());
  17.360 μs (1 allocation: 48 bytes)
```
