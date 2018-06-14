using Base.Test
using LightKDE, Cubature

@testset "Normalisation" begin
    for d in (1, 2, 3)
        data = zeros(d,1)

        for k in (Exponential, Gaussian, Tophat, Triangular)
            z, _ = hcubature_v(fill(-3, d), fill(3, d), abstol = 1e-1) do x,v
                v[:] = pdf(kde(data, kernel = k), x)
            end
            @test z ≈ 1 atol = 0.2
        end
    end
end

@testset "Likelihood cross-validation" begin
    srand(9000)
    data  = randn(2,100)
    k1 = kde(data, cross_validate = true)
    k2 = kde(data, cross_validate = false)

    @test mean(pdf(k1, data)) >= mean(pdf(k2, data))
end

