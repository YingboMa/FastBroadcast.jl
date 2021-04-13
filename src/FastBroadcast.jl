module FastBroadcast

export fast_materialize!

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
        $(bcc.loopheader)
        if $islinear($axes(dst), $(bcc.arrays...))
            @inbounds @simd for i in $eachindex(dst)
                $loopbody_lin
            end
        else
            Base.Cartesian.@nloops $(ndims(dst)) i dst begin
                $loopbody_car
            end
        end
        dst
    end
end

@inline islinear(dst, src) = dst == axes(src)
@inline islinear(dst, src, srcs::Vararg{AbstractArray,K}) where {K} = dst == axes(src) && islinear(dst, srcs...)

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

end
