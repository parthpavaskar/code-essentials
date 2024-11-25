#***********************************************************************
module shockfind
#***********************************************************************


# MATH PACKAGES
using Statistics
using Distributions 
using FFTW, Random
using FITSIO, LsqFit
using Interpolations
using Polynomials
using LinearAlgebra
using StaticArrays

# DATA PACKAGES
using HDF5, DataFrames
using Images
using FortranFiles
using DelimitedFiles

# PLOTTING PACKAGES
using PyPlot
using LaTeXStrings

# QOL PACKAGES
using BenchmarkTools 
using ProgressMeter

# CESSENTIALS
using data_struct
using base_io
using array_functions

#***********************************************************************

export calc_convergence, calc_gradient, get_candidates
export shock_remove, shock_remove_nodens, shock_remove_nodens_meanfill
export shock_remove_v, shock_remove_jump
#***********************************************************************


function calc_convergence(vx_in::Array{T, 3}, vy_in::Array{T, 3}, vz_in::Array{T, 3}) where T
    nx, ny, nz = size(vx_in)
    vx = add_periodic_ghost_cells(vx_in, 1)
    vy = add_periodic_ghost_cells(vx_in, 1)
    vz = add_periodic_ghost_cells(vx_in, 1)
    
    div_u = init_array(T, nx, ny, nz)
    
    for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
        dvx_dx = (vx[i+1, j, k] - vx[i-1, j, k]) / (2)
        dvy_dy = (vy[i, j+1, k] - vy[i, j-1, k]) / (2)
        dvz_dz = (vz[i, j, k+1] - vz[i, j, k-1]) / (2)

        div_u[i, j, k] = dvx_dx + dvy_dy + dvz_dz
        
    end
    return div_u .* (-1)
end


function calc_gradient(ρ_in::Array{T, 3}) where T
    nx, ny, nz = size(ρ_in)
    ρ = add_periodic_ghost_cells(ρ_in, 2)
    
    ∂ρx = init_array(T, nx, ny, nz)
    ∂ρy = init_array(T, nx, ny, nz)
    ∂ρz = init_array(T, nx, ny, nz)
    
#     for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
#         # Second order central finite difference
#         ∂x[i-1, j-1, k-1] = (ρ[i+1, j, k] - ρ[i-1, j, k]) / 2
#         ∂y[i-1, j-1, k-1] = (ρ[i, j+1, k] - ρ[i, j-1, k]) / 2
#         ∂z[i-1, j-1, k-1] = (ρ[i, j, k+1] - ρ[i, j, k-1]) / 2

    for k in 3:nz+2, j in 3:ny+2, i in 3:nx+2
        # Fourth order central finite difference
        ∂ρx[i-2, j-2, k-2] = ( -ρ[i-2, j, k] + 8*ρ[i-1, j, k] - 8*ρ[i+1, j, k] + ρ[i+2, j, k]) / 12
        ∂ρy[i-2, j-2, k-2] = ( -ρ[i, j-2, k] + 8*ρ[i, j-1, k] - 8*ρ[i, j+1, k] + ρ[i, j+2, k]) / 12
        ∂ρz[i-2, j-2, k-2] = ( -ρ[i, j, k-2] + 8*ρ[i, j, k-1] - 8*ρ[i, j, k+1] + ρ[i, j, k+2]) / 12
    end
    
    ∇ρ = sqrt.(∂ρx.^2 .+ ∂ρy.^2 .+ ∂ρz.^2)
    return ∂ρx, ∂ρy, ∂ρz, ∇ρ
end


function get_candidates(ρ::Array{T,3}, mean_times_fac::T) where T
    
    ∂x, ∂y, ∂z, ∇ρ = calc_gradient(ρ)
    n_candidates = findall(>(mean_times_fac * mean(∇ρ)), ∇ρ)
    
    return n_candidates
end


function shock_remove_nodens(arr::Array{T, 3}, n_candidates::Vector{CartesianIndex{3}}, rem_size::Int) where T
    
    arr_new = add_periodic_ghost_cells(arr, rem_size)    
    
    for (iter_ind, candidate_ind) in enumerate(n_candidates)
        
        arr_new[candidate_ind[1] : candidate_ind[1] + 2*rem_size,
            candidate_ind[2] : candidate_ind[2] + 2*rem_size,
            candidate_ind[3] : candidate_ind[3] + 2*rem_size] .= 0
        
    end
    return arr_new[(rem_size+1):(end-rem_size), (rem_size+1):(end-rem_size), (rem_size+1):(end-rem_size)]
end

function shock_remove_nodens_meanfill(arr::Array{T, 3}, n_candidates::Vector{CartesianIndex{3}}, rem_size::Int) where T
    
    arr_new = add_periodic_ghost_cells(arr, rem_size)    
    m_arr = mean(arr)    
    
    for (iter_ind, candidate_ind) in enumerate(n_candidates)
        
        arr_new[candidate_ind[1] : candidate_ind[1] + 2*rem_size,
            candidate_ind[2] : candidate_ind[2] + 2*rem_size,
            candidate_ind[3] : candidate_ind[3] + 2*rem_size] .= m_arr
        
    end
    return arr_new[(rem_size+1):(end-rem_size), (rem_size+1):(end-rem_size), (rem_size+1):(end-rem_size)]
end


function shock_remove(arr::Array{T, 3}, ρ::Array{T, 3}, threshold_grad::T, rem_size::Int) where T
    
    arr_new = add_periodic_ghost_cells(arr, rem_size)    
    
    ∂x, ∂y, ∂z, ∇ρ = calc_gradient(ρ)
    n_candidates = findall(>(threshold_grad), ∇ρ)
    
    for (iter_ind, candidate_ind) in enumerate(n_candidates)
#         candidate_dir = [∂x[candidate_ind], ∂y[candidate_ind], ∂z[candidate_ind]] ./ ∇ρ[candidate_ind]
        
        arr_new[candidate_ind[1] : candidate_ind[1] + 2*rem_size,
            candidate_ind[2] : candidate_ind[2] + 2*rem_size,
            candidate_ind[3] : candidate_ind[3] + 2*rem_size] .= 0
        
    end
    return arr_new[(rem_size+1):(end-rem_size), (rem_size+1):(end-rem_size), (rem_size+1):(end-rem_size)]
end

function shock_remove_v(arr::Array{T, 3}, vx::Array{T, 3}, vy::Array{T, 3}, vz::Array{T, 3}, threshold_conv::T, rem_size::Int) where T
    
    arr_new = add_periodic_ghost_cells(arr, rem_size)    
    
    ∇dotv = calc_convergence(vx, vy, vz)
    n_candidates = findall(>(threshold_conv), ∇dotv)
    
    for (iter_ind, candidate_ind) in enumerate(n_candidates)
#         candidate_dir = [∂x[candidate_ind], ∂y[candidate_ind], ∂z[candidate_ind]] ./ ∇ρ[candidate_ind]
        
        arr_new[candidate_ind[1] : candidate_ind[1] + 2*rem_size,
            candidate_ind[2] : candidate_ind[2] + 2*rem_size,
            candidate_ind[3] : candidate_ind[3] + 2*rem_size] .= 0
        
    end
    return arr_new[(rem_size+1):(end-rem_size), (rem_size+1):(end-rem_size), (rem_size+1):(end-rem_size)]
end

function shock_remove_jump(arr::Array{T, 3}, ρ::Array{T, 3}, threshold_grad::T, threshold_jump::T, rem_size::Int) where T
    
    arr_new = add_periodic_ghost_cells(arr, rem_size)
    ρ_ght = add_periodic_ghost_cells(ρ, 10)
    
    ∂x, ∂y, ∂z, ∇ρ = calc_gradient(ρ)
    n_candidates = findall(>(threshold_grad), ∇ρ)
    rj=[]
    
    for (iter_ind, candidate_ind) in enumerate(n_candidates)
        
        shock_width = 1 / ∇ρ[candidate_ind]
        
        step_x = Int(round(shock_width * ∂x[candidate_ind]))
        step_y = Int(round(shock_width * ∂y[candidate_ind]))
        step_z = Int(round(shock_width * ∂z[candidate_ind]))
        
        i_pre_shock = candidate_ind[1] + 10 - step_x
        j_pre_shock = candidate_ind[2] + 10 - step_y
        k_pre_shock = candidate_ind[3] + 10 - step_z
        
        i_post_shock = candidate_ind[1] + 10 + step_x
        j_post_shock = candidate_ind[2] + 10 + step_y
        k_post_shock = candidate_ind[3] + 10 + step_z
        
        density_jump = maximum([abs(ρ_ght[i_post_shock, j_post_shock, k_post_shock] / ρ_ght[i_pre_shock, j_pre_shock, k_pre_shock]), abs(ρ_ght[i_pre_shock, j_pre_shock, k_pre_shock] / ρ_ght[i_post_shock, j_post_shock, k_post_shock])])
        append!(rj, density_jump)
        
        if density_jump >= threshold_jump
            arr_new[candidate_ind[1] : candidate_ind[1] + 2*rem_size,
            candidate_ind[2] : candidate_ind[2] + 2*rem_size,
            candidate_ind[3] : candidate_ind[3] + 2*rem_size] .= 0
        end
    end
    return arr_new[(rem_size+1):(end-rem_size), (rem_size+1):(end-rem_size), (rem_size+1):(end-rem_size)], rj
end


# function shockfind(conv::Array{T, 3}, grad::Array{T, 3}, conv_threshold::T, threshold_grad::T, Ncells::Int, block=0::Int, quiet=false::Bool, ID="") where T
#     """
#     Function to flag shock candidate cells. Outputs a list of
#     coordinates to be used by extract function.
    
#     Parameters
#     ----------
#         conv : 3d array
#             array of convergences to be searched
#         grad : 3d array
#             array of gradient magnitudes to be
#             searched. shape(grad) must equal shape(conv).
#         conv_threshold : float
#             find_shocks searches for cells with
#             convergence values above this threshold
#         threshold_grad : float
#             find_shocks searches for cells with
#             gradient magnitudes above this threshold
#         Ncells : int
#             max number of cells to flag, find_shocks stops
#             searching beyond this number
#         block : int
#             blocking length around shock candidate
    
#     Returns
#     -------
#         shocklist : list
#             list of locations of shock candidates [list of 3-tuples]
#     """
    


# #     if not quiet:
# #         print ""*{0:s} Searching down to convergence of {1:.3f}. Then for gradients down to {2:.3f}.".format(ID, conv_threshold,threshold_grad)
# #         print ""*{0:s} Search limit set to {1:d} cells.".format(ID, Ncells)
        
#     search_loop = "conv"
#     search_iterator = conv.copy()
#     threshold=conv_threshold
#     mask = np.init_array_like(search_iterator, dtype=np.int64)



#     xmax = np.shape(conv)[0] # x-limit
#     ymax = np.shape(conv)[1] # y-limit
#     zmax = np.shape(conv)[2] # z-limit
    
#     i=0
#     j=0
    
#     x_loc, y_loc, z_loc = np.array([],dtype=np.int64),np.array([],dtype=np.int64),np.array([],dtype=np.int64)
#     cell_conv, cell_grad = np.array([]),np.array([])
    
#     # Search through convergences until threshold or Ncells is reached
#     while search_loop == "conv" or search_loop =="grad":
#         t0 = time.time()
#         j+=1
#         idx_tmp = masked_search(search_iterator, mask)
#         #idx_abs = (idx_tmp[0]+loadx1,idx_tmp[1]+y_low,idx_tmp[2]+z_low) # used to be absolute coordinates, but now this function won't know about it
        
#         mask[idx_tmp]=1 # block out this candidate in the mask
        
        
#         # Break the loop when the search limit is reached
#         if j == Ncells+2:
#             print "*{0:s} Reached shock search limit: {1:d}".format(ID, Ncells)

#             if search_loop == "conv":
#                 output_conv = [
#                     x_loc,
#                     y_loc,
#                     z_loc,
#                     cell_conv,
#                     cell_grad
#                 ]
                
#                 output_grad = [
#                     np.array([]),
#                     np.array([]),
#                     np.array([]),
#                     np.array([]),
#                     np.array([])
#                 ]
#             elif search_loop == "grad":
#                 output_grad = [
#                     x_loc,
#                     y_loc,
#                     z_loc,
#                     cell_conv,
#                     cell_grad
#                 ]
#             break
            
#         # Break the loop when the search threshold is reached
#         elif search_iterator[idx_tmp][0] < threshold:
#             if search_loop=="grad":
#                 if not quiet:
#                     print '*{0:s} grad threshold reached.'.format(ID)

#                 output_grad = [
#                     x_loc,
#                     y_loc,
#                     z_loc,
#                     cell_conv,
#                     cell_grad
#                 ]
#                 break
                
#             else:
#                 print '*{0:s} conv threshold reached, searching through gradients now.'.format(ID)

#                 output_conv = [
#                     x_loc,
#                     y_loc,
#                     z_loc,
#                     cell_conv,
#                     cell_grad
#                 ]

#                 search_iterator=grad.copy()
#                 threshold=threshold_grad
#                 search_loop = "grad"
#                 x_loc = np.array([],dtype=np.int64)
#                 y_loc = np.array([],dtype=np.int64)
#                 z_loc = np.array([],dtype=np.int64)
#                 cell_conv=[]
#                 cell_grad=[]
#                 continue

#         # Ignore candidates that are diverging
#         if conv[idx_tmp][0] <= 0:
#             continue

        
        
        
# #         print '{0:s} #{1:d} ({2:d},{3:d},{4:d}), {5:s}: {6:.3f}, threshold: {7:.3f}, t: {8:.3f} s'.format(ID, i+1, idx_tmp[0][0], idx_tmp[1][0], idx_tmp[2][0], search_loop, search_iterator[idx_tmp][0], threshold, time.time()- t0)
                          
                          
#         # Save shock location, gradient, convergence
#         x_loc=np.append(x_loc, idx_tmp[0][0])
#         y_loc=np.append(y_loc, idx_tmp[1][0])
#         z_loc=np.append(z_loc, idx_tmp[2][0])
#         cell_conv=np.append(cell_conv, conv[idx_tmp][0])
#         cell_grad=np.append(cell_grad, grad[idx_tmp][0])
        
#         i+=1


#         if block > 0:
#             # Block out the surrounding region in the search iterator cube
#             x1 = max(0., idx_tmp[0][0]-block)
#             y1 = max(0., idx_tmp[1][0]-block)
#             z1 = max(0., idx_tmp[2][0]-block)

#             x2 = min(xmax, idx_tmp[0][0]+block+1)
#             y2 = min(ymax, idx_tmp[1][0]+block+1)
#             z2 = min(zmax, idx_tmp[2][0]+block+1)

#             mask[x1:x2, y1:y2, z1:z2] = 1 # block around the candidate in the mask

#     return output_conv, output_grad

##############################################################

end #top
