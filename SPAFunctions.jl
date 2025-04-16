#***********************************************************************
module SPAFunctions
#***********************************************************************

# MATH PACKAGES
using Statistics
using FFTW, Random
using LinearAlgebra
using Images, LsqFit

# CUSTOM PACKAGES
using DataStruct
using BaseIO
using ArrayFunctions
using DecompMHD

#***********************************************************************

export los_integrate 
export get_stokes, get_stokes_w09, get_stokes_fr_emitting_layer
export get_stokes_fr_emitting_layer_sm, get_stokes_fr_foreground_random
export get_sxx, sxx_spap_fit, sxx_full_fit, get_spap_fit, get_full_fit
export get_ξxx, get_yxx, linearize_sxx, s_fluct_array, spa_zhang
export spa_plus, spa_plus_obs
#***********************************************************************

"""
    Module SpaFunctions

Functions for numerical tests of the SPA++ (modified Synchrotron Polarization Analysis)
    perfromed in Pavaskar et al. (2024) [DOI: 10.3847/1538-4357/ad5af5].
"""

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


function get_sxx(I::Matx, Q::Matx, U::Matx)::Tuple{Vect, Vect}
    ϕ_arr = zeros(0)
    sxx_iqd2 = zeros(0)
    ϕ_obs = 0.5 .* atan.(U,Q)
    ϕ_obs_mean = mean(ϕ_obs)
    for ϕ_s in deg2rad.(0:1:180)
        push!(ϕ_arr, ϕ_obs_mean + ϕ_s)
        Q_rot = Q .* cos(2.0 * ϕ_s) - U .* sin(2.0 * ϕ_s)
        IQd2 = (I .+ Q_rot) ./ 2.0
        push!(sxx_iqd2, var(IQd2))
    end
    return ϕ_arr, sxx_iqd2
end


sxx_spap_fit(x, p) = @.(p[1] + p[2]*sin(2*x) + p[3]*sin(4*x) + p[4]*cos(2*x) + p[5]*cos(4*x))
sxx_full_fit(x, p) = @.(p[1] + p[2]*sin(x) + p[3]*sin(2*x) + p[4]*sin(3*x) + p[5]*sin(4*x) + p[6]*cos(x) + p[7]*cos(2*x) + p[8]*cos(3*x) + p[9]*cos(4*x))

function get_spap_fit(ϕ_s::Vect, s_xx::Vect)::Vect
    cfit_xx = curve_fit(sxx_spap_fit, ϕ_s, s_xx, [rand(5)...])
    return cfit_xx.param
end

function get_full_fit(ϕ_s::Vect, s_xx::Vect)::Vect 
    cfit_xx = curve_fit(sxx_full_fit, ϕ_s, s_xx, [rand(9)...])
    return cfit_xx.param   
end

struct SpaPlusParams
    bx_in::Cube
    by_in::Cube
    bz_in::Cube
    bx_in_A::Cube
    by_in_A::Cube
    bz_in_A::Cube
    bx_in_F::Cube
    by_in_F::Cube
    bz_in_F::Cube
    bx_in_S::Cube
    by_in_S::Cube
    bz_in_S::Cube
end

function spa_plus(nres::Integer, params::SpaPlusParams; 
    lower_lim::Integer=20, upper_lim::Integer=80, diff::Integer=10,
    sub_center::Bool=true, sub_i::Integer=1, sub_j::Integer=1,
    k_inj::Integer=2, gam88::Bool=false)::Tuple{Array{Float64,3}, Array{Float64,3}, Array{Float64,3}}

    gamma_r = collect(lower_lim:diff:upper_lim)
    
    # Add θ_λ = 88 deg instead of 90 deg to remove Alfven problem case
    if gam88
        push!(gamma_r, 88)
    end

    sub_r = div(nres, k_inj)

    bx_ins = zeros(Float64, 4, nres, nres, nres)
    by_ins = zeros(Float64, 4, nres, nres, nres)
    bz_ins = zeros(Float64, 4, nres, nres, nres)

    phi_s = zeros(Float64, 181, 4, length(gamma_r))
    fpar_fn = zeros(Float64, 5, 4, length(gamma_r))

    modes_str = ("B", "A", "F", "S")

    bx_ins[1, :,:,:], by_ins[1, :,:,:], bz_ins[1, :,:,:] = params.bx_in, params.by_in, params.bz_in
    bx_ins[2, :,:,:], by_ins[2, :,:,:], bz_ins[2, :,:,:] = params.bx_in_A, params.by_in_A, params.bz_in_A
    bx_ins[3, :,:,:], by_ins[3, :,:,:], bz_ins[3, :,:,:] = params.bx_in_F, params.by_in_F, params.bz_in_F
    bx_ins[4, :,:,:], by_ins[4, :,:,:], bz_ins[4, :,:,:] = params.bx_in_S, params.by_in_S, params.bz_in_S 

    for (θ, val) in enumerate(gamma_r) 
        print("Starting iteration θ_λ = ", val)
        flush(stdout)

        θ_λ = deg2rad(val)
        for mode in 1:4
            bx_rot, by_rot, bz_rot = vector_rotation(bx_ins[mode,:,:,:], 
            by_ins[mode,:,:,:], bz_ins[mode,:,:,:], 0, θ_λ, 0)

            ii_map, qq_map, uu_map = get_stokes(bx_rot, by_rot)
            # ii_map, qq_map, uu_map = get_stokes_fr_foreground_random(bx_rot, by_rot, 50)
            
            ii_spt = sub_grid(ii_map, sub_r, center=sub_center, grid_i=sub_i, grid_j=sub_j)
            qq_spt = sub_grid(qq_map, sub_r, center=sub_center, grid_i=sub_i, grid_j=sub_j)
            uu_spt = sub_grid(uu_map, sub_r, center=sub_center, grid_i=sub_i, grid_j=sub_j)

            phi_s[:,mode,θ], sxx_fn[:,mode,θ] = get_sxx(ii_spt, qq_spt, uu_spt)
            fpar_fn[:,mode,θ] = get_spap_fit(phi_s[:,mode,θ], sxx_fn[:,mode,θ])
        end
        println("  .... done")
        flush(stdout)
    end
    return phi_s, sxx_fn, fpar_fn
end

function spa_plus_obs(ii_map::Matx, qq_map::Matx, uu_map::Matx, spotsize::Integer)::Cube
    nres = size(ii_map)[1]

    overlap_spot = div(spotsize, 2)
    scan_grid = collect(1:overlap_spot:nres-spotsize)
    phi_s = zeros(Float64, 181, length(scan_grid), length(scan_grid))
    fpar_fn = zeros(Float64, 5, length(scan_grid), length(scan_grid))

    for i in 1:length(scan_grid), j in 1:length(scan_grid)
        ii_spt = ii_map[scan_grid[i]:scan_grid[i]+spotsize, scan_grid[j]:scan_grid[j]+spotsize]
        qq_spt = qq_map[scan_grid[i]:scan_grid[i]+spotsize, scan_grid[j]:scan_grid[j]+spotsize]
        uu_spt = uu_map[scan_grid[i]:scan_grid[i]+spotsize, scan_grid[j]:scan_grid[j]+spotsize]

        phi_s, sxx_fn = get_sxx(ii_spt, qq_spt, uu_spt)
        fpar_fn[:,i,j] = get_spap_fit(phi_s, sxx_fn)
    end
    return fpar_fn
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
    rxx_fit = curve_fit(s_model_zhang, Δϕ_s, s_fluct, [rand(2)...])

    rxx = (rxx_fit.param[1] / rxx_fit.param[2]) * (2*sin(2*δθ) / sin(4*δθ))
    return rxx, ξ, y
end


end # top
