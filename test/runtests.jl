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
end
