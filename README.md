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
