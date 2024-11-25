#***********************************************************************
module SpaFunctions
#***********************************************************************

# MATH PACKAGES
using Statistics
using FFTW, Random
using LinearAlgebra
using Images

# CUSTOM PACKAGES
using .DataStruct
using .BaseIO
using .ArrayFunctions
using .DecompMHD

#***********************************************************************

export los_integrate 
export get_stokes, get_stokes_w09, get_stokes_fr_emitting_layer
export get_stokes_fr_emitting_layer_sm, get_stokes_fr_foreground_random
export get_sxx, sxx_spap_fit, sxx_full_fit, get_spap_fit, get_full_fit
export get_ξxx, get_yxx, linearize_sxx, s_fluct_array, spa_zhang
export spa_zhang

#***********************************************************************


function los_integrate(a::Cube; dims=3)
    return reshape(sum(a, dims=dims), size(a)[1:2])
end


function get_stokes(bi::Cube, bj::Cube)::Tuple{Matx, Matx, Matx}
    I = los_integrate(bi.^2.0 .+ bj.^2.0)
    Q = los_integrate(bi.^2.0 .- bj.^2.0)
    U = los_integrate(2.0 .* bi .* bj)
    return I, Q, U
end


function get_stokes_w09(bi::Cube, bj::Cube; p=3::Float64)::Tuple{Matx, Matx, Matx}
    electron_term = (bi.^2.0 .+ bj.^2.0).^((p-3.0)/4.0)
    I = los_integrate(electron_term.* (bi.^2.0 .+ bj.^2.0))
    Q = los_integrate(electron_term.* (bi.^2.0 .- bj.^2.0))
    U = los_integrate(electron_term.* (2.0 .* bi .* bj))
    return I, Q, U
end


function get_stokes_fr_emitting_layer(bi::Cube, bj::Cube, bk::Cube, dens::Cube, phi_deg::Integer)::Tuple{Matx, Matx, Matx}
    nx, ny, nz = size(bi)
    I = zeros(nx,ny)
    Q = zeros(nx,ny)
    U = zeros(nx,ny)
    phi = deg2rad(phi_deg) / nz
    bmean = sqrt((mean(bi))^2 + (mean(bj))^2 + (mean(bk))^2)
    dmean = mean(dens)
    for j in 1:ny, i in 1:nx
        Qs1=0
        Us1=0
        for k in 1:nz
            phi_fr = (bk[i,j,k] / bmean) * (dens[i,j,k] / dmean) * (phi)
            I[i,j] += (bi[i,j,k]^2 + bj[i,j,k]^2)
            Qs0 = (bi[i,j,k]^2 - bj[i,j,k]^2) + Qs1
            Us0 = (2 * bi[i,j,k] * bj[i,j,k]) + Us1
            Qs1 = Qs0 * cos(phi_fr) + Us0 * sin(phi_fr)
            Us1 = Us0 * cos(phi_fr) - Qs0 * sin(phi_fr)
        end
        Q[i,j] = Qs1
        U[i,j] = Us1
    end
    return I, Q, U
end

function get_stokes_fr_emitting_layer_sm(dn::Cube, bi::Cube, bj::Cube, bk::Cube, nu::Float64, L1::Float64)::Tuple{Matx, Matx, Matx}
    lambda = 3e+8 / nu
    nx, ny, nz = size(bi)
    I = zeros(nx,ny)
    Q = zeros(nx,ny)
    U = zeros(nx,ny)
    # L1 = 50
    for j in 1:ny, i in 1:nx   
        phi = 0.0
        for k in 1:nz
            phi += lambda^2 * dn[i,j,k] * bk[i,j,k] * (L1 / nz)
            I[i,j] += (bi[i,j,k]^2 + bj[i,j,k]^2)
            Qs0 = (bi[i,j,k]^2 - bj[i,j,k]^2)
            Us0 = (2 * bi[i,j,k] * bj[i,j,k])
            Q[i,j] += Qs0 * cos(phi) + Us0 * sin(phi)
            U[i,j] += Us0 * cos(phi) - Qs0 * sin(phi)
        end
    end
    return I, Q, U

end


function get_stokes_fr_foreground_random(bi::Cube, bj::Cube, fr_std::Integer)::Tuple{Matx, Matx, Matx}
    nx, ny, nz = size(bi)
    Q = init_array(Float64, nx, ny)
    U = init_array(Float64, nx, ny)
    I = los_integrate(bi.^2.0 .+ bj.^2.0)
    Q_tmp = los_integrate(bi.^2.0 .- bj.^2.0)
    U_tmp = los_integrate(2.0 .* bi .* bj)

    Random.seed!(121354565663)
    fr_norm = Normal(0, fr_std)
    fr_map_deg = rand(fr_norm, nx, ny)
    fr_map_rad = deg2rad.(fr_map_deg)
    for jj in 1:ny, ii in 1:nx
        Q[ii, jj] = Q_tmp[ii, jj] * cos(fr_map_rad[ii, jj]) + U_tmp[ii, jj] * sin(fr_map_rad[ii, jj])
        U[ii, jj] = U_tmp[ii, jj] * cos(fr_map_rad[ii, jj]) - Q_tmp[ii, jj] * sin(fr_map_rad[ii, jj])
    end
        
    return I, Q, U
end


function get_sxx(I::Matx, Q::Matx, U::Matx)::Tuple{Vect, Vect, Float64}
    ϕ_arr = zeros(0)
    sxx_iqd2 = zeros(0)
    ϕ_obs = 0.5 .* atan.(U,Q)
    ϕ_obs_mean = mean(ϕ_in)
    for ϕ_s in deg2rad.(0:1:180)
        push!(ϕ_arr, ϕ_obs_mean + ϕ_s)
        Q_rot = Q .* cos(2.0 * ϕ_s) - U .* sin(2.0 * ϕ_s)
        IQd2 = (I .+ Q_rot) ./ 2.0
        push!(sxx_iqd2, var(IQd2))
    end
    return ϕ_arr, sxx_iqd2, ϕ_obs_mean
end


sxx_spap_fit(x, p) = @.(p[1] + p[2]*sin(2*x) + p[3]*sin(4*x) + p[4]*cos(2*x) + p[5]*cos(4*x))
sxx_full_fit(x, p) = @.(p[1] + p[2]*sin(x) + p[3]*sin(2*x) + p[4]*sin(3*x) + p[5]*sin(4*x) + p[6]*cos(x) + p[7]*cos(2*x) + p[8]*cos(3*x) + p[9]*cos(4*x))

function get_spap_fit(ϕ_s::Vect, s_xx::Vect)::Vect
    cfit_xx = curve_fit(sxx_new_fit, ϕ_s, s_xx, [rand(5)...])
    return cfit_xx.param
end
end

function get_full_fit(ϕ_s::Vect, s_xx::Vect)::Vect 
    cfit_xx = curve_fit(sxx_full_fit, ϕ_s, s_xx, [rand(9)...])
    return cfit_xx.param   
end













function get_ξxx(s_xx::Vect)::Float64

    return (maximum(s_xx)-minimum(s_xx))./(maximum(s_xx)+minimum(s_xx))
end

function get_yxx(ϕ_s::Vect, s_xx::Vect)::Float64

    s_max=maximum(s_xx)
    s_min=minimum(s_xx)
    
    s_denom = 0.5 .* (s_max + s_min)
    
    s_2 = 0
    s_4 = 0
    Δϕ = abs(diff(ϕ_s)[1])
    
    for i in 1:length(ϕ_s)-1
        s_2 += 0.5 .* (s_xx[i] * sin(2.0.*ϕ_s[i]) .+ s_xx[i+1] * sin(2.0.*ϕ_s[i+1])) * Δϕ
        s_4 += 0.5 .* (s_xx[i] * sin(4.0.*ϕ_s[i]) .+ s_xx[i+1] * sin(4.0.*ϕ_s[i+1])) * Δϕ
    end
    return sqrt(s_2^2 + s_4^2) / s_denom
end

linear_sxx(x, p) = (p[1].*sin.(x).*sin.(x) .+ p[2]) .* cos.(x) .* cos.(x) .+ p[3]

function linearize_sxx(ϕ_s::Vect, s_xx::Vect)::Vect
    cfit = curve_fit(linear_sxx, ϕ_s, s_xx, [rand(3)...])
    return s_xx .- cfit.param[3]
end


function s_fluct_array(ϕ_s::Vect, s_xx::Vect; θ_mean=π/2::Float64)::Vect
    Δϕ = abs(diff(ϕ_s)[1])
    Δϕ_s = 0.01:0.01:pi/4
    s_f = zeros(length(Δϕ_s))
    
    for j in 1:length(Δϕ_s), i in 1:length(ϕ_s)-1
            dϕ = 0.5 * (ϕ_s[i] + ϕ_s[i+1]) - θ_mean
            if abs(dϕ) < Δϕ_s
                s_f[j] += 0.5 * (s_xx[i] + s_xx[i+1]) * Δϕ
            end
    end
    return s_f
end


s_model_zhang(x, p) = (4.0 .* (p[1] .+ 4.0 .* p[2]) .* x .- 8.0 .* p[2] .* sin.(2.0.*x) .- p[1] .* sin.(4.0.*x)) ./ 16

function spa_zhang(I::Vect, Q::Vect, U::Vect; 
    δθ=mean(0.5.*atan.(U, Q))::Float64, θ_mean=π/2::Float64,
    Δϕ_s=0.01:0.01:π/4::Vect)::Tuple{Float64, Float64, Float64}

    ϕ_s, s_xx = get_sxx(I, Q, U)

    ξ = get_ξxx(s_xx)
    y = get_yxx(ϕ_s, s_xx)

    s_xx = linearize_sxx!(ϕ_s, s_xx)
    s_fluct = s_fluct_array(ϕ_s, s_xx, Δϕ_s, θ_mean=θ_mean)
    rxx_fit = curve_fit(s_model_zhang, Δϕ_s, s_fluct ,[rand(2)...])

    rxx = (rxx_fit.param[1] / rxx_fit.param[2]) * (2*sin(2*δθ) / sin(4*δθ))
    return rxx, ξ, y
end
