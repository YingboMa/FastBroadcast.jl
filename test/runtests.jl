using FastBroadcast
using Test

@testset "FastBroadcast.jl" begin
    x, y = [1,2,3,4], [5,6,7,8]
    dst = zeros(Int, 4)
    bc = Broadcast.Broadcasted(+, (Broadcast.Broadcasted(*, (x, y)), x, y, x, y, x, y))
    bcref = copy(bc)
    @test FastBroadcast.fast_materialize!(dst, bc) == bcref
    dest = similar(bcref);
    @test (@.. dest = (x*y) + x + y + x + y + x + y) == bcref
    @test (@.. (x*y) + x + y + x + y + x + y) == bcref
    @test (@.. dest += (x*y) + x + y + x + y + x + y) ≈ 2bcref
    @test (@.. dest -= (x*y) + x + y + x + y + x + y) ≈ bcref
    @test (@.. dest *= (x*y) + x + y + x + y + x + y) ≈ abs2.(bcref)
    @test (@.. (x*y) + x + y + x*(3,4,5,6) + y + x * (1,) + y + 3) ≈ (@. (x*y) + x + y + x*(3,4,5,6) + y + x * (1,) + y + 3)
    A = rand(4,4);
    @test (@.. A * y' + x) ≈ (@. A * y' + x)
end
