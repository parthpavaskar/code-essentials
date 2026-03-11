#***********************************************************************
module DataStruct
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

#***********************************************************************

export set_resolution, set_radiation 
export calc_Bmean_2D, calc_Bmean_3D, calc_vposition
export calc_pol_frac, calc_stokes!, calc_pol_correction
export get_dϕ_pol, get_dϕ_pol_vvf

#***********************************************************************

"""
    Module DCFGSAFunctions

Functions for numerical testing of the DCF method with GSA polarization
    perfromed in Pavaskar et al. (2023) [DOI: 10.1093/mnras/stad1237].
"""

#***********************************************************************


function set_resolution(bb::Cube; n_resize_los::Integer, n_resize_pos::Integer, n_rot::Integer, n_gsa::Integer)
    
    ni_in, nj_in, nk_in = size(bb)
    ni = nj = n_resize_pos
    nk = n_resize_los

    nrot = n_rot
    ngsa = n_gsa

    return ni_in, nj_in, nk_in, ni, nj, nk, nrot, ngsa 
end


function set_radiation(ϕ_0_deg, θ_0_deg, pump_norm)
    
    pumping = pump_norm

    ϕ_0 = deg2rad(ϕ_0_deg)
    θ_0 = deg2rad(θ_0_deg)
    
    return ϕ_0, θ_0, pumping    
end


function calc_Bmean_2D(bi::Cube, bj::Cube, ni::Integer, nj::Integer, nk::Integer)
    
    bi_mean = sum(bi) / (ni * nj * nk)
    bj_mean = sum(bj) / (ni * nj * nk)
    bt_mean = sqrt(bi_mean^2 + bj_mean^2)
    
    return bt_mean
end


function calc_Bmean_3D(bi::Cube, bj::Cube, bk::Cube, ni::Integer, nj::Integer, nk::Integer)
    
    bi_mean = sum(bi) / (ni * nj * nk)
    bj_mean = sum(bj) / (ni * nj * nk)
    bk_mean = sum(bk) / (ni * nj * nk)
    bt_mean = sqrt(bi_mean^2 + bj_mean^2 + bk_mean^2)
    
    return bt_mean
end



function calc_obf_phi(vec_i::Float64, vec_j::Float64, vec_k::Float64, vec0_i::Float64, vec0_j::Float64, vec0_k::Float64)
    return acos(vec_i*vec0_i + vec_j*vec0_j / sqrt((vec_i^2 + vec_j^2) * (vec0_i^2 + vec0_j^2)))
end


function calc_vposition(vec_i::Float64, vec_j::Float64, vec_k::Float64)
    phi = acos(vec_i / sqrt(vec_i^2 + vec_j^2))
    theta = acos(vec_k / sqrt(vec_i^2 + vec_j^2 + vec_k^2))
    
    return phi, theta
end


function calc_pol_frac(gsa_array::Cube, θ_b::Float64, ϕ_b::Float64, θ_0::Float64, ngsa::Integer=size(gsa_array)[1])
    θ_b_int = round(Int, (θ_b * (ngsa - 1) / π)) + 1
    ϕ_b_int = round(Int, (ϕ_b * (ngsa - 1) / π)) + 1
    θ_0_int = round(Int, (θ_0 * (ngsa - 1) / π)) + 1
    
    pol_tot = gsa_array[ϕ_b_int,θ_0_int,θ_b_int]
    return pol_tot, sign(pol_tot), 1.
end


function calc_stokes!(i_stk::Float64, q_stk::Float64, u_stk::Float64, dn::Float64, ϕ_b::Float64, pol::Float64)
    i_step = i_stk + dn
    q_step = q_stk + dn * cos(2*ϕ_b) * pol
    u_step = u_stk + dn * sin(2*ϕ_b) * pol
    
    return i_step, q_step, u_step
end


function calc_pol_correction(q_stokes::Float64, u_stokes::Float64)        
            
    if q_stokes > 0 && u_stokes >= 0
        ϕp = (0.5 * atan(u_stokes / q_stokes))
        
    elseif q_stokes > 0 && u_stokes < 0
        ϕp = (0.5 * atan(u_stokes / q_stokes)) + π
        
    elseif q_stokes < 0
        ϕp = (0.5 * atan(u_stokes / q_stokes)) + π/2
        
    elseif q_stokes == 0
        ϕp = π/4
    end
    
   return ϕp
end


function get_dϕ_pol(pol_map::Any, np0::Integer=36)

    ni, nj, nrot, nrot = size(pol_map)

    dph = init_array(Float64, nrot, nrot)

    @showprogress for dθ in 1:25, dϕ in 1:25
        var_norm = init_array(Float64, np0+1)
        for iϕ in 0:np0

            p0_obs = deg2rad(iϕ * (180/np0))
            var_sum = 0.

            for ij in 1:nj, ii in 1:ni

                phi_rot = pol_map[ii,ij, dϕ, dθ]
                var_sum += (phi_rot - p0_obs)^2
            end

            var_norm[iϕ+1] = var_sum / (ni * nj - 1)
        end

        dph[dϕ,dθ] = minimum(sqrt.(var_norm))
    end

    return dph
end



function get_dϕ_pol_vvf(pol_map::Any, np0::Integer=36, filter_r::Integer=1, filter_t::Float64=10.0)

    ni, nj, nrot, nrot = size(pol_map)

    dph = init_array(Float64, nrot, nrot)
    
    @showprogress for dθ in 1:nrot, dϕ in 1:nrot
        
        var_sum = init_array(Float64, np0+1)
        
        for ij in 1+filter_r:nj-filter_r, ii in 1+filter_r:ni-filter_r

            vicinity_mean = mean(pol_map[ii-filter_r:ii+filter_r, ij-filter_r:ij+filter_r, dϕ,dθ])
            vicinity_sd = sqrt(var(pol_map[ii-filter_r:ii+filter_r, ij-filter_r:ij+filter_r, dϕ,dθ]))
            
            fparam = (pol_map[ni,nj, dϕ,dθ] - vicinity_mean) / vicinity_sd
                
            if fparam < filter_t

                for iϕ in 0:np0

                    p0_obs = deg2rad(iϕ * (180/np0))
                    
                    phi_rot = pol_map[ii,ij, dϕ, dθ]
                    var_sum[iϕ+1] += (phi_rot - p0_obs)^2

                end
            end
        end
        
        dph[dϕ,dθ] = minimum(sqrt.(var_sum ./ (ni * nj - 1)))
    end

    return dph
end


function get_dϕ_pol_cstd(pol_map::Any)

    _, _, nrot, nrot = size(pol_map)

    dph = init_array(Float64, nrot, nrot)

    for dθ in 1:nrot, dϕ in 1:nrot
        pol_map_or = pol_map[:,:, dϕ,dθ]
        dph[dϕ,dθ] = DefIO.cstd(pol_map_or)
    end

    return dph
end

end #top