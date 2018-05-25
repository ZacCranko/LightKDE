module LightKDE

using Distributions, MLDataUtils

abstract type AbstractKernel end

eval_kernel(k::AbstractKernel, x, y) = k(x - y)
eval_kernel(k::ContinuousMultivariateDistribution, x, y) = exp(sum(logpdf.(k, x - y)))

lp_lognormalisation(inv_r, d::Int, p) = -d*(log(2) - log(inv_r) + lgamma(1 + 1/p)) + lgamma(1 + d/p)

mutable struct Ball <: AbstractKernel
    band::Real
    p::Real

    Ball(band, p = 2.0) = new(band, p) # all kernels must have a single argument constructor
end

(b::Ball)(x) = norm(x / b.band, b.p) <= 1 ? 1 : 0
_lognormalisation(k::Ball, ::Val{d}) where d = lp_lognormalisation(k.band, d, k.p)

mutable struct Tophat <: AbstractKernel
    band::Real
end

(t::Tophat)(x) = norm(x / t.band, Inf) <= 1 ? 1 : 0
_lognormalisation(t::Tophat, ::Val{d}) = 1/(2pi)^(d-1) = lp_lognormalisation(t.band, d, Inf)
Gaussian(band)    = Distributions.Normal(0, band)
Exponential(band) = Distributions.Laplace(0, band)

_lognormalisation(::Distributions.Normal, ::Val{d}) where d = (1-d) * (log(2) + log(pi))
(g::Normal)(x) = exp(sum(logpdf.(Normal(0, g.band), x)))

mutable struct Triangular <: AbstractKernel
    band::Real
end

(t::Triangular)(x) = norm(x / t.band, 1) <= 1 ? 1 - nx : 0


lognormalisation(k, d::Int) = _lognormalisation(k, Val{d}())

struct KernelDensity{d} <: ContinuousMultivariateDistribution
    kernel::K
    data::AbstractMatrix
    weights::AbstractVector
    
    function KernelDensity(kernel::AbstractKernel, data::AbstractArray, weights::AbstractVector)
        size(data, ndims(data)) == length(weights) || throw("must have the same number of data ($(size(data, 2))) and weights ($(length(weights)))")
        all(w >= 0 for w in weights)     || throw("weights must all be nonnegative")
        s = sum(weights)
        if !isapprox(s, 1) 
            weights ./= s
        end
        d = size(data, 1)
        return new{d}(kernel, data, weights)
    end
end

KernelDensity(kernel::AbstractKernel, data::AbstractArray) = KernelDensity(kernel, data, fill(1/size(data, ndims(data)), size(data, ndims(data))))

function Distributions.pdf(k::KernelDensity{d}, x::Union{Real, AbstractVector})
    size(x, 1) == size(k.data, 1) || throw("incorrectly sized input with dimension $(size(x, 1)); expected $(size(k.data, 1))")
    density = zero(Float64)
    @inbounds for i in Base.OneTo(size(k.data, ndims(k.data)))
        density += eval_kernel(k.kernel, x, k.data[:,i]) * k.weights[i]
    end
    return exp(log(density) - lognormalisation(k, d))
end

function Distributions.pdf(k::KernelDensity, x::AbstractMatrix)
    n = size(x, 2)
    density = Vector{Float64}(n)
    Threads.@threads for i in 1:n
        @inbounds density[i] = pdf(k, x[:,i])
    end
    return density
end

Distributions.logpdf(k::KernelDensity, x) = log.(pdf(k, x))

using MLDataUtils

scott(d, n)::Float64     = n^(-1/(d + 4))
silverman(d, n)::Float64 = (n*(d + 2)/4)^(-1/(d + 4))

bandwidth!(k::AbstractKernel, band) = (k.band = band; return k)
bandwidth!(k::KernelDensity,  band) = bandwidth!(k.kernel, band)

function kde(data, weights = fill(1/size(data,2), size(data,2)); 
             band::Union{Real, typeof(scott), typeof(silverman)} = scott, 
             kernel::Union{K, Type{K}} = Gaussian, 
             cross_validate = false, 
             folds = 10, 
             δ = 2) where K <: AbstractKernel

    band::Real             = isa(band, Real) ? band : band(size(data)...)
    kernel::AbstractKernel = isa(kernel, AbstractKernel) ? kernel : kernel(band)

    k = KernelDensity(kernel, data, weights)

    if cross_validate
        results    = zeros(Float64, folds)
        bandwidths = linspace(max(eps(), kernel.band - δ), kernel.band + δ, folds)
        for (i, b) in enumerate(bandwidths) 
            for (train_idx, valid_idx) in kfolds(size(data, 2), folds)
                bandwidth!(kernel, b)
                k           = KernelDensity(kernel, data[:, train_idx], weights[train_idx])
                results[i] += sum(logpdf(k, data[:, valid_idx]))
            end
        end
        _, mx = findmax(results)
        if mx == 1 || mx == folds 
            # if likelihood is maximised by either the first or last bandwidth in the search
            warn("a better model may be achieved by increasing δ")
        end
        k.kernel.band = bandwidths[mx]
    end
    
    return KernelDensity(kernel, data, weights)
end


export Gaussian, Normal, Exponential, Laplace, Ball, Tophat, Laplace, KernelDensity, Exponential, KernelDensity, kde, pdf, scott, silverman
 

end # module