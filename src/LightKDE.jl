module LightKDE

using Distributions, MLDataUtils

const Kernel = ContinuousUnivariateDistribution

function eval_kernel(k::Kernel, x::T, y::U) where {T,U}
    p = zero(promote_type(eltype(T), eltype(U)))
    n = length(x)
    @inbounds for i in Base.OneTo(n)
        p += logpdf(k, x[i] - y[i])
    end
    return exp(p)
end

Tophat(band)      = Distributions.Uniform(-band, +band)
Gaussian(band)    = Distributions.Normal(0, band)
Exponential(band) = Distributions.Exponential(band)
Triangular(band)  = Distributions.SymTriangularDist(0, band)

struct Epanechnikov <: ContinuousUnivariateDistribution
    b::Real
end

Distributions.pdf(e::Epanechnikov,    x::Real) = abs(x/e.b) < 1 ? 3/(4b)*(1 - (x/e.b)^2) : 0
Distributions.logpdf(e::Epanechnikov, x::Real) = abs(x/e.b) < 1 ? log(3) + log1p(-(x/e.b)^2) - log(4) - log(e.b) : -Inf

struct Cosine <: ContinuousUnivariateDistribution
    b::Real
end

Distributions.pdf(d::Cosine,    x::Real) = abs(x/e.b) <= 1 ? pi/(4d.b) * cospi(x/(4d.b)) : 0
Distributions.logpdf(d::Cosine, x::Real) = abs(x/d.b) <= 1 ? log(pi) - log(4) - log(d.b) + log(cospi(x/(4d.b))) : -Inf

bandwidth(::Distributions.Uniform, band)            = Distributions.Uniform(-band, +band)
bandwidth(::Distributions.Normal, band)             = Distributions.Normal(0, band)
bandwidth(::Distributions.Exponential, band)        = Distributions.Exponential(band)
bandwidth(::Distributions.SymTriangularDist, band)  = Distributions.SymTriangularDist(0, band)
bandwidth(::Epanechnikov, band)                     = Epanechnikov(band)
bandwidth(::Cosine, band)                           = Cosine(band)
supported_kernels = (Epanechnikov, Gaussian, Tophat, Exponential, Cosine, Triangular)

struct KernelDensity <: ContinuousMultivariateDistribution
    kernel::Kernel
    data::AbstractArray
    weights::AbstractVector
    
    function KernelDensity(kernel, data, weights)
        size(data, ndims(data)) == length(weights) || throw("must have the same number of data ($(size(data, 2))) and weights ($(length(weights)))")
        all(w >= 0 for w in weights) || throw("weights must all be nonnegative")
        s = sum(weights)
        if !isapprox(s, 1) 
            weights ./= s
        end
        return new(kernel, data, weights)
    end
end

function KernelDensity(kernel::Kernel, data::AbstractArray)
    weights = fill(1/size(data, ndims(data)), size(data, ndims(data)))
    KernelDensity(kernel, data, weights)
end

function Distributions.pdf(k::KernelDensity, x::AbstractVector)
    size(x, 1) == size(k.data, 1) || throw("incorrectly sized input with dimension $(size(x, 1)); expected $(size(k.data, 1))")
    density = zero(Float64)
    @inbounds for i in Base.OneTo(size(k.data, ndims(k.data)))
        density += eval_kernel(k.kernel, x, k.data[:,i]) * k.weights[i]
    end
    return density
end

function Distributions.pdf(k::KernelDensity, x::AbstractMatrix)
    n = size(x, 2)
    density = Vector{Float64}(n)
    Threads.@threads for i in 1:n
        density[i] = pdf(k, x[:,i])
    end
    return density
end

function Base.rand(k::KernelDensity)
    i = rand(Distributions.Categorical(k.weights))
    return rand(k.kernel, size(k.kernel, 2)) + k.data[:,i]
end

function Base.rand!(k::KernelDensity, result::AbstractMatrix{T}) where T<:Real
    return reshape(rand(k.kernel, length(result)), size(result)) 
end

Distributions.logpdf(k::KernelDensity, x::AbstractMatrix) = log.(pdf(k, x))

using MLDataUtils

scott(d, n)::Float64     = n^(-1/(d + 4))
silverman(d, n)::Float64 = (n*(d + 2)/4)^(-1/(d + 4))

function kde(data, weights = fill(1/size(data, ndims(data)), size(data, ndims(data))); 
             band::Union{Real, typeof(scott), typeof(silverman)} = scott, 
             kernel = Gaussian, 
             cross_validate = false, 
             folds = 10, 
             δ = 2)

    band::Real = isa(band, Real) ? band : band(size(data)...)
    kernel::Kernel = isa(kernel, Kernel) ? kernel : kernel(band)

    k = KernelDensity(kernel, data, weights)

    if !cross_validate
        return k
    else
        results    = zeros(Float64, folds)
        densities  = Vector{KernelDensity}(folds)
        bandwidths = vcat(linspace(max(0, band - δ), band + δ, folds)[2:end], band)
        for (i, b) in enumerate(bandwidths) 
            for (train_idx, valid_idx) in kfolds(size(data, 2), folds)
                densities[i] = KernelDensity(bandwidth(kernel, b), data[:, train_idx], weights[train_idx])
                results[i]  += sum(logpdf(k, data[:, valid_idx]))
            end
        end
        _, mx = findmax(results)
        return densities[mx]
    end
end


export Epanechnikov, Gaussian, Normal, Exponential, Triangular, Tophat, KernelDensity, kde, pdf, scott, silverman
 

end # module