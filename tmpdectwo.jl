#***********************************************************************
module tmpdectwo
#***********************************************************************

# MATH PACKAGES
using Statistics
using FFTW, Random
using LinearAlgebra
using ProgressMeter

# CUSTOM PACKAGES
using DataStruct
using BaseIO
using ArrayFunctions
using FLoops

#***********************************************************************

export decomposition_kmtwo


#***********************************************************************


function decomposition_kmtwo(
    iv::Cube, jv::Cube, kv::Cube, ib::Cube, jb::Cube, kb::Cube,
    ρ::Cube, β::Float64; ε::Float64=(0.5/size(iv)[1]^2))
    
    # Get the dimensions of the input cubes
    nx, ny, nz = size(iv)

    # Calculate the mean density
    ρ0 = mean(ρ)

    # Calculate the mean magnetic field unit vector
    B0 = [mean(ib), mean(jb), mean(kb)]
    Bhat = normalize(B0)

    # Calculate v_A and c_s
    v_A = norm(B0) / sqrt(ρ0)
    c_s = sqrt(β * v_A^2 / 2)

    # Calculate α = β * (γ / 2)
    α = β / 2.0

    # Perform FFT and shift the zero-frequency component to the center
    norm_factor = nx * ny * nz
    dn_fft = fftshift(fft(ρ)) ./ norm_factor
    iv_fft, jv_fft, kv_fft = fft_components(iv, jv, kv, norm_factor)
    ib_fft, jb_fft, kb_fft = fft_components(ib, jb, kb, norm_factor)

    vv_fft_out = zeros(ComplexF64, 3, size(iv_fft)..., 3)
    bb_fft_out = zeros(ComplexF64, 3, size(ib_fft)..., 3)

    @floop for k in 1:nz, j in 1:ny, i in 1:nx

        # Assign desity, vecocity, and magnetic field Fourier vector at (i,j,k)
        d_fft = dn_fft[i, j, k]
        v_fft = [iv_fft[i, j, k], jv_fft[i, j, k], kv_fft[i, j, k]]
        b_fft = [ib_fft[i, j, k], jb_fft[i, j, k], kb_fft[i, j, k]]
                
        # Get RHS of the eigenvector equation
        rhs = [d_fft, v_fft..., b_fft...]
        
        # Calculate the unit wave vector
        kx = i - div(nx, 2) - 1
        ky = j - div(ny, 2) - 1
        kz = k - div(nz, 2) - 1
        kk = [kx, ky, kz]
        khat = normalize(kk)

        # Avoid zero wave vectors
        if !iszero(norm(kk))
            
            # Remove potential (divergent) part of B
            b_pot = dot(khat, b_fft) * khat
            
            # Calculate cosθ
            cosθ = dot(khat, Bhat)

            # Calculate non-negative D
            # D = maximum([0.0, (1 + α)^2 - (4 * α * cosθ^2)])
            D = (1 + α)^2 - (4 * α * cosθ^2)

            # Calculate the wave speeds
            c_wave_a = v_A * cosθ
            c_wave_s = v_A * sqrt(0.5 * ((1.0 + α) - sqrt(D)))
            c_wave_f = v_A * sqrt(0.5 * ((1.0 + α) + sqrt(D)))

            if abs(cosθ) < ε  # Perpendicular propagation (k_para=0)
                
                # Get the velocity basis vectors
                ξv_mode_A = normalize(cross(khat, Bhat))
                ξv_mode_S = Bhat
                ξv_mode_F = khat

                # Assign eigenvectors
                out_fft_Ap = zeros(7)
                out_fft_Am = zeros(7)
                
                eigvec_Anp1 = [0.0, ξv_mode_A..., zeros(3)...]
                eigvec_Anp2 = [0.0, zeros(3)..., ξv_mode_A...]

                out_fft_Sp = zeros(7)
                out_fft_Sm = zeros(7)
                
                eigvec_Snp1 = [0.0, ξv_mode_S..., zeros(3)...]
                eigvec_Snp2 = [ρ0, zeros(3)..., (-α*B0)...]

                ξb_mode_F = normalize(- cross(khat, cross(ξv_mode_F, B0)))
                ξd_mode_F = ρ0 * dot(khat, ξv_mode_F) ./ c_wave_f
                
                eigvec_Fp = [ξd_mode_F, ξv_mode_F..., ξb_mode_F...]
                eigvec_Fm = [-ξd_mode_F, ξv_mode_F..., -ξb_mode_F...]
                
                p1 = 4 + argmax(eigvec_Fp[5:7])
                p2 = 4 + argmax(eigvec_Anp2[5:7])
                
                p2 += (p1 == p2 ? (p2 < 7 ? 1 : -1) : 0)
                pcase = 1
                out_fft_Anp1, out_fft_Anp2, out_fft_Snp1, out_fft_Snp2, out_fft_Fp, out_fft_Fm = 
                    solve_linear_system(rhs, eigvec_Anp1, eigvec_Anp2, eigvec_Snp1, eigvec_Snp2, eigvec_Fp, eigvec_Fm, p1, p2, pcase)

                

            elseif 1.0 - abs(cosθ) < ε # Parallel propagation (k_perp=0)

                # Generate random vector perpendicular to B0
                rand_vec = Random.rand(Random.GLOBAL_RNG, 3)
                ξv_mode_A = normalize(cross(Bhat, rand_vec))
                
                if α >= 1.0
                    ξv_mode_F = khat
                    ξv_mode_S = normalize(cross(ξv_mode_A, ξv_mode_F))
                else
                    ξv_mode_S = khat
                    ξv_mode_F = normalize(cross(ξv_mode_A, ξv_mode_S))
                end
                
                # Assign eigenvectors
                ξb_mode_A = normalize(- cross(khat, cross(ξv_mode_A, B0)) ./ c_wave_a)
                eigvec_Ap = [0.0, ξv_mode_A..., ξb_mode_A...]
                eigvec_Am = [0.0, ξv_mode_A..., -ξb_mode_A...]
                
                out_fft_Anp1 = zeros(7)
                out_fft_Anp2 = zeros(7)

                ξb_mode_S = normalize(- cross(khat, cross(ξv_mode_S, B0)) ./ c_wave_s)
                ξd_mode_S = ρ0 * dot(khat, ξv_mode_S) ./ c_wave_s
                
                eigvec_Sp = [ξd_mode_S, ξv_mode_S..., ξb_mode_S...]
                eigvec_Sm = [-ξd_mode_S, ξv_mode_S..., -ξb_mode_S...]
                
                out_fft_Snp1 = zeros(7)
                out_fft_Snp2 = zeros(7)

                ξb_mode_F = normalize(- cross(khat, cross(ξv_mode_F, B0)) ./ c_wave_f)
                ξd_mode_F = ρ0 * dot(khat, ξv_mode_F) ./ c_wave_f

                eigvec_Fp = [ξd_mode_F, ξv_mode_F..., ξb_mode_F...]
                eigvec_Fm = [-ξd_mode_F, ξv_mode_F..., -ξb_mode_F...]

                p1 = 4 + argmax(eigvec_Ap[5:7])
                p2 = 4 + argmax(eigvec_Sp[5:7])
                p3 = 4 + argmax(eigvec_Fp[5:7])
                
                p2 = (p1 == p2 ? p3 : p2)
                p2 += (p1 == p2 ? (p2 < 7 ? 1 : -1) : 0)

                pcase = 2
                out_fft_Ap, out_fft_Am, out_fft_Sp, out_fft_Sm, out_fft_Fp, out_fft_Fm = 
                    solve_linear_system(rhs, eigvec_Ap, eigvec_Am, eigvec_Sp, eigvec_Sm, eigvec_Fp, eigvec_Fm, p1, p2, pcase)
            
            else # Oblique propagation
                
                k_para_hat = Bhat
                k_perp_hat = normalize(kk - Bhat * cosθ)
                
                eigvec = zeros(6, 6)
                
                # Get the basis vectors
                ξv_mode_A = normalize(cross(khat, Bhat))
                

                para_fact = sqrt(1.0 - cosθ^2) * cosθ * c_s^2
                perp_fact_S = c_wave_s^2 - (cosθ * c_s)^2
                perp_fact_F = c_wave_f^2 - (cosθ * c_s)^2
                
                ξv_mode_S = normalize(para_fact * k_para_hat + perp_fact_S * k_perp_hat)
                ξv_mode_F = normalize(para_fact * k_para_hat + perp_fact_F * k_perp_hat)

                # Assign eigenvectors
                ξb_mode_A = normalize(- cross(khat, cross(ξv_mode_A, B0)) ./ c_wave_a)
                eigvec_Ap = [0.0, ξv_mode_A..., ξb_mode_A...]
                eigvec_Am = [0.0, ξv_mode_A..., -ξb_mode_A...]
                
                out_fft_Anp1 = zeros(7)
                out_fft_Anp2 = zeros(7)

                ξb_mode_S = normalize(- cross(khat, cross(ξv_mode_S, B0)) ./ c_wave_s)
                ξd_mode_S = ρ0 * dot(khat, ξv_mode_S) ./ c_wave_s
                
                eigvec_Sp = [ξd_mode_S, ξv_mode_S..., ξb_mode_S...]
                eigvec_Sm = [-ξd_mode_S, ξv_mode_S..., -ξb_mode_S...]
                
                out_fft_Snp1 = zeros(7)
                out_fft_Snp2 = zeros(7)

                ξb_mode_F = normalize(- cross(khat, cross(ξv_mode_F, B0)) ./ c_wave_f)
                ξd_mode_F = ρ0 * dot(khat, ξv_mode_F) ./ c_wave_f

                eigvec_Fp = [ξd_mode_F, ξv_mode_F..., ξb_mode_F...]
                eigvec_Fm = [-ξd_mode_F, ξv_mode_F..., -ξb_mode_F...]
                
                p1 = 4 + argmax(eigvec_Ap[5:7])
                p2 = 4 + argmax(eigvec_Sp[5:7])
                p3 = 4 + argmax(eigvec_Fp[5:7])
                
                p2 = (p1 == p2 ? p3 : p2)
                p2 += (p1 == p2 ? (p2 < 7 ? 1 : -1) : 0)

                pcase = 3
                out_fft_Ap, out_fft_Am, out_fft_Sp, out_fft_Sm, out_fft_Fp, out_fft_Fm = 
                    solve_linear_system(rhs, eigvec_Ap, eigvec_Am, eigvec_Sp, eigvec_Sm, eigvec_Fp, eigvec_Fm, p1, p2, pcase)
            end
            
            vv_fft_out[1,i,j,k,:] = out_fft_Ap[2:4] + out_fft_Am[2:4] + out_fft_Anp1[2:4] + out_fft_Anp2[2:4]
            vv_fft_out[2,i,j,k,:] = out_fft_Sp[2:4] + out_fft_Sm[2:4] + out_fft_Snp1[1:3] + out_fft_Snp2[2:4]
            vv_fft_out[3,i,j,k,:] = out_fft_Fp[2:4] + out_fft_Fm[2:4]

            bb_fft_out[1,i,j,k,:] = out_fft_Ap[5:7] + out_fft_Am[5:7] + out_fft_Anp1[5:7] + out_fft_Anp2[5:7]
            bb_fft_out[2,i,j,k,:] = out_fft_Sp[5:7] + out_fft_Sm[5:7] + out_fft_Snp1[5:7] + out_fft_Snp2[5:7]
            bb_fft_out[3,i,j,k,:] = out_fft_Fp[5:7] + out_fft_Fm[5:7]

            sumk = out_fft_Ap + out_fft_Am + out_fft_Sp + out_fft_Sm + out_fft_Fp + out_fft_Fm + 
            out_fft_Anp1 + out_fft_Anp2 + out_fft_Snp1 + out_fft_Snp2
            
            residue = (rhs .- sumk)# + b_pot
        end # k=0
    end # ijk

    iv_out_A, jv_out_A, kv_out_A = [inverse_fft_component(vv_fft_out, 1, i) for i in 1:3]
    ib_out_A, jb_out_A, kb_out_A = [inverse_fft_component(bb_fft_out, 1, i) for i in 1:3]

    iv_out_S, jv_out_S, kv_out_S = [inverse_fft_component(vv_fft_out, 2, i) for i in 1:3]
    ib_out_S, jb_out_S, kb_out_S = [inverse_fft_component(bb_fft_out, 2, i) for i in 1:3]

    iv_out_F, jv_out_F, kv_out_F = [inverse_fft_component(vv_fft_out, 3, i) for i in 1:3]
    ib_out_F, jb_out_F, kb_out_F = [inverse_fft_component(bb_fft_out, 3, i) for i in 1:3]

    return_array = [iv_out_A, jv_out_A, kv_out_A, ib_out_A, jb_out_A, kb_out_A,
                    iv_out_S, jv_out_S, kv_out_S, ib_out_S, jb_out_S, kb_out_S,
                    iv_out_F, jv_out_F, kv_out_F, ib_out_F, jb_out_F, kb_out_F]

    return return_array
end # function

function inverse_fft_component(data, mode_idx, component_idx)
    return real(ifft(ifftshift(data[mode_idx, :, :, :, component_idx])))
end

function solve_linear_system(rhs::CmVect, A1::Vect, A2::Vect, 
    S1::Vect, S2::Vect, F1::Vect, F2::Vect, p1::Integer, p2::Integer, pcase::Integer)::Tuple{CmVect, CmVect, CmVect, CmVect, CmVect, CmVect}
    
    b_inds = [5, 6, 7]
    p2 = (abs(round(A1[p1], digits=4)) == abs(round(A1[p2], digits=4)) ? setdiff(b_inds, [p1, p2])[1] : p2)

    if iszero(A1[p1]) && iszero(A2[p1]) && iszero(S1[p1]) && iszero(S2[p1]) && iszero(F1[p1]) && iszero(F2[p1])
        p1 = setdiff(b_inds, [p1, p2])[1]
    end

    if iszero(A1[p2]) && iszero(A2[p2]) && iszero(S1[p2]) && iszero(S2[p2]) && iszero(F1[p2]) && iszero(F2[p2])
        p2 = setdiff(b_inds, [p1, p2])[1]
    end

    # Stack all eigenvectors as columns
    eigenvecs = [A1, A2, S1, S2, F1, F2]
    eigenvec = hcat([vcat(vec[1:4], vec[p1], vec[p2]) for vec in eigenvecs]...)
    
    # println(A1)
    # println(A2)
    # println(S1)
    # println(S2)
    # println(F1)
    # println(F2, p1, p2)
    # println(eigenvec)
    # flush(stdout)

    # if abs(det(lu(complex(eigenvec)))) < 1e-12
    #     println("error")
    #     println(eigenvec)
    #     flush(stdout)
    # end

    # Solve and multiply back to get decomposed components
    coeffs = lu(complex(eigenvec)) \ [rhs[1:4]..., rhs[p1], rhs[p2]]
    
    # println(coeffs[1] .* eigenvecs[1])
    # lu_fact = lu(complex(eigenvec))
    # if abs(det(lu_fact)) < 1e-12
    #     # Matrix is singular, return zeros
    #     coeffs = zeros(ComplexF64, 6)
    # else
    #     coeffs = lu_fact \ [rhs[1:4]..., rhs[p1], rhs[p2]]
    # end
    
    return tuple((coeffs[i] .* eigenvecs[i] for i in 1:6)...)
end

function fft_components(iv, jv, kv, normalization)
    iv_fft = fftshift(fft(iv)) ./ normalization
    jv_fft = fftshift(fft(jv)) ./ normalization
    kv_fft = fftshift(fft(kv)) ./ normalization
    return iv_fft, jv_fft, kv_fft
end


#***********************************************************************
end # module
