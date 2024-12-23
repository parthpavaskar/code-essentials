#***********************************************************************
module tmpdec
#***********************************************************************

# MATH PACKAGES
using Statistics
using FFTW, Random
using LinearAlgebra

# CUSTOM PACKAGES
using DataStruct
using BaseIO
using ArrayFunctions

#***********************************************************************

export decomposition_km


#***********************************************************************


function decomposition_km(
    iv::Cube, jv::Cube, kv::Cube, ib::Cube, jb::Cube, kb::Cube,
    ρ::Cube, β::Float64; ε::Float64=(0.5/size(iv)[1]^2)
    )::Tuple{Cube, Cube, Cube, Cube, Cube, Cube}
    
    # Get the dimensions of the input cubes
    nx, ny, nz = size(iv)

    # Calculate the mean density
    ρ0 = mean(ρ)

    # Calculate the mean magnetic field unit vector
    B0 = [mean(ib), mean(jb), mean(kb)]
    Bhat = normalize(B0)

    # Calculate mean velocity vector
    v0 = [mean(iv), mean(jv), mean(kv)]

    # Calculate v_A and c_s
    v_A = norm(B0) / sqrt(ρ0)
    c_s = sqrt(β * v_A^2 / 2)

    # Calculate α = β * (γ / 2)
    α = β / 2.0

    # Perform FFT and shift the zero-frequency component to the center
    dn_fft = fftshift(fft(ρ)) ./ (nx * ny * nz)
    
    # Perform FFT and shift the zero-frequency component to the center
    iv_fft = fftshift(fft(iv)) ./ (nx * ny * nz)
    jv_fft = fftshift(fft(jv)) ./ (nx * ny * nz)
    kv_fft = fftshift(fft(kv)) ./ (nx * ny * nz)

    # Perform FFT and shift the zero-frequency component to the center
    ib_fft = fftshift(fft(ib)) ./ (nx * ny * nz)
    jb_fft = fftshift(fft(jb)) ./ (nx * ny * nz)
    kb_fft = fftshift(fft(kb)) ./ (nx * ny * nz)

    vv_fft_out = zeros(ComplexF64, 3, size(iv_fft)..., 3)

    bb_fft_out = zeros(ComplexF64, 3, size(ib_fft)..., 3)

    for k in 1:nz, j in 1:ny, i in 1:nx

        # Assign desity Fourier
        d_fft = dn_fft[i, j, k]

        # Assign vecocity Fourier vector at (i,j,k)
        v_fft = [iv_fft[i, j, k], jv_fft[i, j, k], kv_fft[i, j, k]]

        # Assign magnetic field Fourier vector at (i,j,k)
        b_fft = [ib_fft[i, j, k], jb_fft[i, j, k], kb_fft[i, j, k]]

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
            
            # Calculate the dot product of khat and Bhat (cosθ)
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
                
                eigvec_Anp1 = [0.0, ξv_mode_A..., 0.0, 0.0, 0.0]
                eigvec_Anp2 = [0.0, 0.0, 0.0, 0.0, ξv_mode_A...]

                out_fft_Sp = zeros(7)
                out_fft_Sm = zeros(7)
                
                eigvec_Snp1 = [0.0, ξv_mode_S..., 0.0, 0.0, 0.0]
                eigvec_Snp2 = [ρ0, 0.0, 0.0, 0.0, (-α*B0)...]

                ξb_mode_F = - cross(khat, cross(ξv_mode_F, B0))
                ξd_mode_F = ρ0 * dot(khat, ξv_mode_F) ./ c_wave_f
                
                eigvec_Fp = [ξd_mode_F, ξv_mode_F..., ξb_mode_F...]
                eigvec_Fm = [-ξd_mode_F, ξv_mode_F..., -ξb_mode_F...]
                
                # Get RHS of the eigenvector equation
                rhs = [v_fft..., b_fft...]
                
                p1 = 4 + argmax(eigvec_Fp[5:7])
                p2 = 4 + argmax(eigvec_Ann2[5:7])
                
                p2 += (p1 == p2 ? (p2 < 7 ? 1 : -1) : 0)

                out_fft_Anp1, out_fft_Anp2, out_fft_Snp1, out_fft_Snp2, out_fft_Fp, out_fft_Fm = 
                    solve_linear_system(rhs, eigvec_Anp1, eigvec_Anp2, eigvec_Snp1, eigvec_Snp2, eigvec_Fp, eigvec_Fm, p1, p2)

                

            elseif 1.0 - abs(cosθ) < ε # Parallel propagation (k_perp=0)

                
                # Generate random vector perpendicular to B0
                rand_vec = rand(3)
                ξv_mode_A = normalize(cross(Bhat, rand_vec))
                
                if α >= 1.0
                    ξv_mode_F = khat
                    ξv_mode_S = normalize(cross(ξv_mode_A, ξv_mode_F))
                else
                    ξv_mode_S = khat
                    ξv_mode_F = normalize(cross(ξv_mode_A, ξv_mode_S))
                end
                
                # Assign eigenvectors
                ξb_mode_A = - cross(khat, cross(ξv_mode_A, B0)) ./ c_wave_a
                eigvec_Ap = [0.0, ξv_mode_A..., ξb_mode_A...]
                eigvec_Am = [0.0, ξv_mode_A..., -ξb_mode_A...]
                
                out_fft_Anp1 = zeros(7)
                out_fft_Anp2 = zeros(7)

                ξb_mode_S = - cross(khat, cross(ξv_mode_S, B0)) ./ c_wave_s
                ξd_mode_S = ρ0 * dot(khat, ξv_mode_S) ./ c_wave_s
                
                eigvec_Sp = [ξd_mode_S, ξv_mode_S..., ξb_mode_S...]
                eigvec_Sm = [-ξd_mode_S, ξv_mode_S..., -ξb_mode_S...]
                
                out_fft_Snp1 = zeros(7)
                out_fft_Snp2 = zeros(7)


                ξb_mode_F = - cross(khat, cross(ξv_mode_F, B0)) ./ c_wave_f
                ξd_mode_F = ρ0 * dot(khat, ξv_mode_F) ./ c_wave_f

                eigvec_Fp = [ξd_mode_F, ξv_mode_F..., ξb_mode_F...]
                eigvec_Fm = [-ξd_mode_F, ξv_mode_F..., -ξb_mode_F...]
                
                # Get RHS of the eigenvector equation
                rhs = [d_fft, v_fft..., b_fft...]

                p1 = 4 + argmax(eigvec_Ap[5:7])
                p2 = 4 + argmax(eigvec_Sp[5:7])
                p3 = 4 + argmax(eigvec_Fp[5:7])
                
                p2 = (p1 == p2 ? p3 : p2)
                p2 += (p1 == p2 ? (p2 < 7 ? 1 : -1) : 0)

                out_fft_Ap, out_fft_Am, out_fft_Sp, out_fft_Sm, out_fft_Fp, out_fft_Fm = 
                    solve_linear_system(rhs, eigvec_Ap, eigvec_Am, eigvec_Sp, eigvec_Sm, eigvec_Fp, eigvec_Fm, p1, p2)
            
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
                ξb_mode_A = - cross(khat, cross(ξv_mode_A, B0)) ./ c_wave_a
                eigvec_Ap = [0.0, ξv_mode_A..., ξb_mode_A...]
                eigvec_Am = [0.0, ξv_mode_A..., -ξb_mode_A...]
                
                out_fft_Anp1 = zeros(7)
                out_fft_Anp2 = zeros(7)

                ξb_mode_S = - cross(khat, cross(ξv_mode_S, B0)) ./ c_wave_s
                ξd_mode_S = ρ0 * dot(khat, ξv_mode_S) ./ c_wave_s
                
                eigvec_Sp = [ξd_mode_S, ξv_mode_S..., ξb_mode_S...]
                eigvec_Sm = [-ξd_mode_S, ξv_mode_S..., -ξb_mode_S...]
                
                out_fft_Snp1 = zeros(7)
                out_fft_Snp2 = zeros(7)


                ξb_mode_F = - cross(khat, cross(ξv_mode_F, B0)) ./ c_wave_f
                ξd_mode_F = ρ0 * dot(khat, ξv_mode_F) ./ c_wave_f

                eigvec_Fp = [ξd_mode_F, ξv_mode_F..., ξb_mode_F...]
                eigvec_Fm = [-ξd_mode_F, ξv_mode_F..., -ξb_mode_F...]
                
                # Get RHS of the eigenvector equation
                rhs = [v_fft..., b_fft...]
                
                p1 = 4 + argmax(eigvec_Ap[5:7])
                p2 = 4 + argmax(eigvec_Sp[5:7])
                p3 = 4 + argmax(eigvec_Fp[5:7])
                
                p2 = (p1 == p2 ? p3 : p2)
                p2 += (p1 == p2 ? (p2 < 7 ? 1 : -1) : 0)

                out_fft_Ap, out_fft_Am, out_fft_Sp, out_fft_Sm, out_fft_Fp, out_fft_Fm = 
                    solve_linear_system(rhs, eigvec_Ap, eigvec_Am, eigvec_Sp, eigvec_Sm, eigvec_Fp, eigvec_Fm, p1, p2)
            end

            vv_fft_out[1,i,j,k,:] = out_fft_Ap[1:3] + out_fft_Am[1:3] + out_fft_Anp1[1:3] + out_fft_Anp2[1:3]
            vv_fft_out[2,i,j,k,:] = out_fft_Sp[1:3] + out_fft_Sm[1:3] + out_fft_Snp1[1:3] + out_fft_Snp2[1:3]
            vv_fft_out[3,i,j,k,:] = out_fft_Fp[1:3] + out_fft_Fm[1:3]

            bb_fft_out[1,i,j,k,:] = out_fft_Ap[4:6] + out_fft_Am[4:6] + out_fft_Anp1[4:6] + out_fft_Anp2[4:6]
            bb_fft_out[2,i,j,k,:] = out_fft_Sp[4:6] + out_fft_Sm[4:6] + out_fft_Snp1[4:6] + out_fft_Snp2[4:6]
            bb_fft_out[3,i,j,k,:] = out_fft_Fp[4:6] + out_fft_Fm[4:6]

            sumk = out_fft_Ap + out_fft_Am + out_fft_Sp + out_fft_Sm + out_fft_Fp + out_fft_Fm + 
            out_fft_Anp1 + out_fft_Anp2 + out_fft_Snp1 + out_fft_Snp2
            
            residue = (rhs .- sumk) + b_pot
        end # k=0
    end # ijk

    # Perform inverse FFT and shift back the zero-frequency component
    iv_out_A = real(ifft(ifftshift(vv_fft_out[1,:,:,1])))
    jv_out_A = real(ifft(ifftshift(vv_fft_out[1,:,:,2])))
    kv_out_A = real(ifft(ifftshift(vv_fft_out[1,:,:,3])))
    ib_out_A = real(ifft(ifftshift(bb_fft_out[1,:,:,1])))
    jb_out_A = real(ifft(ifftshift(bb_fft_out[1,:,:,2])))
    kb_out_A = real(ifft(ifftshift(bb_fft_out[1,:,:,3])))

    iv_out_S = real(ifft(ifftshift(vv_fft_out[2,:,:,1])))
    jv_out_S = real(ifft(ifftshift(vv_fft_out[2,:,:,2])))
    kv_out_S = real(ifft(ifftshift(vv_fft_out[2,:,:,3])))
    ib_out_S = real(ifft(ifftshift(bb_fft_out[2,:,:,1])))
    jb_out_S = real(ifft(ifftshift(bb_fft_out[2,:,:,2])))
    kb_out_S = real(ifft(ifftshift(bb_fft_out[2,:,:,3])))

    iv_out_F = real(ifft(ifftshift(vv_fft_out[3,:,:,1])))
    jv_out_F = real(ifft(ifftshift(vv_fft_out[3,:,:,2])))
    kv_out_F = real(ifft(ifftshift(vv_fft_out[3,:,:,3])))
    ib_out_F = real(ifft(ifftshift(bb_fft_out[3,:,:,1])))
    jb_out_F = real(ifft(ifftshift(bb_fft_out[3,:,:,2])))
    kb_out_F = real(ifft(ifftshift(bb_fft_out[3,:,:,3])))

    return_array = [iv_out_A, jv_out_A, kv_out_A, ib_out_A, jb_out_A, kb_out_A,
                    iv_out_S, jv_out_S, kv_out_S, ib_out_S, jb_out_S, kb_out_S,
                    iv_out_F, jv_out_F, kv_out_F, ib_out_F, jb_out_F, kb_out_F]

    return return_array
end # function



function solve_linear_system(rhs::CmVect, A1::Vect, A2::Vect, 
    S1::Vect, S2::Vect, F1::Vect, F2::Vect, p1::Integer, p2::Integer)::Tuple{CmVect, CmVect, CmVect, CmVect, CmVect, CmVect}
    
    eigenvec = zeros(6, 6)
    
    eigenvec[1:4,1] = A1[1:4]
    eigenvec[5,1] = A1[4+p1]
    eigenvec[6,1] = A1[4+p2]

    eigenvec[1:4,2] = A2[1:4]
    eigenvec[5,2] = A2[4+p1]
    eigenvec[6,2] = A2[4+p2]

    eigenvec[1:4,3] = S1[1:4]
    eigenvec[5,3] = S1[4+p1]
    eigenvec[6,3] = S1[4+p2]
    
    eigenvec[1:4,4] = S2[1:4]
    eigenvec[5,4] = S2[4+p1]
    eigenvec[6,4] = S2[4+p2]

    eigenvec[1:4,5] = F1[1:4]
    eigenvec[5,5] = F1[4+p1]
    eigenvec[6,5] = F1[4+p2]

    eigenvec[1:4,6] = F2[1:4]
    eigenvec[5,6] = F2[4+p1]
    eigenvec[6,6] = F2[4+p2]

    # Solve the eigenvector equation
    LinearAlgebra.LAPACK.gesv!(complex(eigenvec), rhs)

    # Alfven mode
    out_fft_Ap = rhs[1] * A1
    out_fft_Am = rhs[2] * A2
    
    # Slow mode
    out_fft_Sp = rhs[3] * S1
    out_fft_Sm = rhs[4] * S2

    # Fast mode
    out_fft_Fp = rhs[5] * F1
    out_fft_Fm = rhs[6] * F2

    return out_fft_Ap, out_fft_Am, out_fft_Sp, out_fft_Sm, out_fft_Fp, out_fft_Fm
end




function decomposition_A_m2(
    iv::Cube, jv::Cube, kv::Cube, 
    B0::Vect, β::Float64; 
    ε::Float64=(0.5/size(iv)[1]^2), ρ0::Float64=1.0)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube}
    
    # Get the dimensions of the input cubes
    nx, ny, nz = size(iv)

    # Calculate the mean magnetic field unit vector
    Bhat = normalize(B0)
    
    # Calculate v_A and c_s
    v_A = norm(B0) / sqrt(ρ0)
    c_s = sqrt(β * v_A^2 / 2)
    
    # Calculate α = β * (γ / 2)
    α = β / 2.0

    # Perform FFT and shift the zero-frequency component to the center
    iv_fft = fftshift(fft(iv))
    jv_fft = fftshift(fft(jv))
    kv_fft = fftshift(fft(kv))

    # Initialize the Fourier output arrays
    vv_fft_out = zeros(typeof(1.0 + im), size(iv_fft)..., 3)
    bb_fft_out = zeros(typeof(1.0 + im), size(iv_fft)..., 3)

    for k in 1:nz, j in 1:ny, i in 1:nx

        # Assign vecocity Fourier vector at (i,j,k)
        v_fft = [iv_fft[i, j, k], jv_fft[i, j, k], kv_fft[i, j, k]]

        # Calculate the unit wave vector
        kx = i - div(nx, 2) - 1
        ky = j - div(ny, 2) - 1
        kz = k - div(nz, 2) - 1
        kk = [kx, ky, kz]
        khat = normalize(kk)

        # Avoid zero wave vectors
        if !iszero(norm(kk))
            # Calculate the dot product of khat and Bhat (cosθ)
            cosθ = dot(khat, Bhat)

            # Alfven mode wave speed
            c_wave = v_A * cosθ

            # Induction equation denominator
            # ind_denom = c_wave * v_A * sqrt(ρ0)
            
            # Remove k_perp = 0 case
            if 1.0 - abs(cosθ) > ε
                #= 
                Get the Alfven mode basis vector
                ξ_Alf = k x B
                =#
                ξ_mode = normalize(cross(khat, Bhat))

                # Get projected Fourier velocity vector
                v_fft_out = dot(v_fft, ξ_mode) .* ξ_mode

                if !iszero(c_wave)
                    # Get direction of magnetic field component
                    b_fft_hat = normalize(cross(khat, cross(v_fft_out, B0)))

                    # Get magnitude of magnetic field component
                    b_magnitude = norm(v_fft_out)

                    # Assign the decomposed magnetic field components
                    bb_fft_out[i,j,k,:] = b_magnitude * b_fft_hat
                    end
                
                # Assign the decomposed velocity components
                vv_fft_out[i,j,k,:] = v_fft_out

            end
        end
    end

    # Perform inverse FFT and shift back the zero-frequency component
    iv_out = real(ifft(ifftshift(vv_fft_out[:,:,:,1])))
    jv_out = real(ifft(ifftshift(vv_fft_out[:,:,:,2])))
    kv_out = real(ifft(ifftshift(vv_fft_out[:,:,:,3])))

    ib_out = real(ifft(ifftshift(bb_fft_out[:,:,:,1])))
    jb_out = real(ifft(ifftshift(bb_fft_out[:,:,:,2])))
    kb_out = real(ifft(ifftshift(bb_fft_out[:,:,:,3])))

    return iv_out, jv_out, kv_out, ib_out, jb_out, kb_out
end



function decomposition_F_m2(
    iv::Cube, jv::Cube, kv::Cube, 
    B0::Vect, β::Float64;
    ε::Float64=(0.5/size(iv)[1]^2), ρ0::Float64=1.0)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube}
    
    # Get the dimensions of the input cubes
    nx, ny, nz = size(iv)

    # Calculate the mean magnetic field unit vector
    Bhat = normalize(B0)
    
    # Calculate v_A and c_s
    v_A = norm(B0) / sqrt(ρ0)
    c_s = sqrt(β * v_A^2 / 2)
    
    # Calculate α = β * (γ / 2)
    α = β / 2.0

    # Perform FFT and shift the zero-frequency component to the center
    iv_fft = fftshift(fft(iv))
    jv_fft = fftshift(fft(jv))
    kv_fft = fftshift(fft(kv))

    # Initialize the Fourier output arrays
    vv_fft_out = zeros(typeof(1.0 + im), size(iv_fft)..., 3)
    bb_fft_out = zeros(typeof(1.0 + im), size(iv_fft)..., 3)

    for k in 1:nz, j in 1:ny, i in 1:nx
        
        # Assign vecocity Fourier vector at (i,j,k)
        v_fft = [iv_fft[i, j, k], jv_fft[i, j, k], kv_fft[i, j, k]]
        
        # Calculate the unit wave vector
        kx = i - div(nx, 2) - 1
        ky = j - div(ny, 2) - 1
        kz = k - div(nz, 2) - 1
        kk = [kx, ky, kz]
        khat = normalize(kk)

        # Avoid zero wave vectors
        if !iszero(norm(kk))
            # Calculate the dot product of khat and Bhat (cosθ)
            cosθ = dot(khat, Bhat)

            # Fast mode wave speed
            Δ2 = (1.0 + α)^2 - (4.0 * α * cosθ^2)
            c_wave = v_A * sqrt(0.5 * ((1.0 + α) + sqrt(Δ2)))

            # Induction equation denominator
            # ind_denom = c_wave * v_A * sqrt(ρ0)
            
            D = (1.0 + α)^2 - (4.0 * α * cosθ^2)
            prefactor_para = - 1.0 + α + sqrt(D)
            prefactor_perp = 1.0 + α + sqrt(D)

            # Remove k_perp = 0 case    
            if 1.0 - abs(cosθ) > ε
                #= 
                Get the fast mode basis vector
                ξ_fast ∝ (- 1 + α + sqrt(D)) * k_para + 
                    (1 + α + sqrt(D)) * k_perp
                =#
                k_para = dot(kk, Bhat) .* Bhat
                k_perp = kk .- k_para
                ξ_mode = normalize((prefactor_para .* k_para) .+ (prefactor_perp .* k_perp))

                # Get projected Fourier velocity vector
                v_fft_out = dot(v_fft, ξ_mode) .* ξ_mode

                if !iszero(c_wave)
                    # Get direction of magnetic field component
                    b_fft_hat = normalize(cross(khat, cross(v_fft_out, B0)))

                    # Get magnitude of magnetic field component
                    b_magnitude = norm(B0) * norm(v_fft_out) * norm(cross(Bhat, ξ_mode)) / c_wave

                    # Assign the decomposed magnetic field components
                    bb_fft_out[i,j,k,:] = b_magnitude * b_fft_hat
                end
                
                # Assign the decomposed velocity components
                vv_fft_out[i,j,k,:] = v_fft_out 
            
            end
        end
    end

    # Perform inverse FFT and shift back the zero-frequency component
    iv_out = real(ifft(ifftshift(vv_fft_out[:,:,:,1])))
    jv_out = real(ifft(ifftshift(vv_fft_out[:,:,:,2])))
    kv_out = real(ifft(ifftshift(vv_fft_out[:,:,:,3])))

    ib_out = real(ifft(ifftshift(bb_fft_out[:,:,:,1])))
    jb_out = real(ifft(ifftshift(bb_fft_out[:,:,:,2])))
    kb_out = real(ifft(ifftshift(bb_fft_out[:,:,:,3])))

    return iv_out, jv_out, kv_out, ib_out, jb_out, kb_out
end



function decomposition_S_m2(
    iv::Cube, jv::Cube, kv::Cube, 
    B0::Vect, β::Float64;
    ε::Float64=(0.5/size(iv)[1]^2), ρ0::Float64=1.0)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube}
    
    # Get the dimensions of the input cubes
    nx, ny, nz = size(iv)
    
    # Calculate the mean magnetic field unit vector
    Bhat = normalize(B0)
    
    # Calculate v_A and c_s
    v_A = norm(B0) / sqrt(ρ0)
    c_s = sqrt(β * v_A^2 / 2)
    
    # Calculate α = β * (γ / 2)
    α = β / 2.0

    # Perform FFT and shift the zero-frequency component to the center
    iv_fft = fftshift(fft(iv))
    jv_fft = fftshift(fft(jv))
    kv_fft = fftshift(fft(kv))

    # Initialize the Fourier output arrays
    vv_fft_out = zeros(typeof(1.0 + im), size(iv_fft)..., 3)
    bb_fft_out = zeros(typeof(1.0 + im), size(iv_fft)..., 3)

    for k in 1:nz, j in 1:ny, i in 1:nx
        
        # Assign vecocity Fourier vector at (i,j,k)
        v_fft = [iv_fft[i, j, k], jv_fft[i, j, k], kv_fft[i, j, k]]
        
        # Calculate the unit wave vector
        kx = i - div(nx, 2) - 1
        ky = j - div(ny, 2) - 1
        kz = k - div(nz, 2) - 1
        kk = [kx, ky, kz]
        khat = normalize(kk)

        # Avoid zero wave vectors
        if !iszero(norm(kk))
            # Calculate the dot product of khat and Bhat (cosθ)
            cosθ = dot(khat, Bhat)

            # Slow mode wave speed
            Δ2 = (1.0 + α)^2 - (4.0 * α * cosθ^2)
            c_wave = v_A * sqrt(0.5 * ((1.0 + α) - sqrt(Δ2)))

            # Induction equation denominator
            # ind_denom = c_wave * v_A * sqrt(ρ0)
            
            D = (1 + α)^2 - (4 * α * cosθ^2)
            prefactor_para = - 1 + α - sqrt(D)
            prefactor_perp = 1 + α - sqrt(D)

            # Remove k_perp = 0 case
            if 1.0 - abs(cosθ) > ε
                #= 
                Get the slow mode basis vector
                ξ_fast ∝ (- 1 + α - sqrt(D)) * k_para + 
                    (1 + α - sqrt(D)) * k_perp
                =#
                k_para = dot(kk, Bhat) .* Bhat
                k_perp = kk .- k_para
                ξ_mode = normalize((prefactor_para .* k_para) .+ (prefactor_perp .* k_perp))

                # Get projected Fourier velocity vector
                v_fft_out = dot(v_fft, ξ_mode) .* ξ_mode

                if !iszero(c_wave)
                    # Get direction of magnetic field component
                    b_fft_hat = normalize(cross(khat, cross(v_fft_out, B0)))

                    # Get magnitude of magnetic field component
                    b_magnitude = norm(B0) * norm(v_fft_out) * norm(cross(Bhat, ξ_mode)) / c_wave

                    # Assign the decomposed magnetic field components
                    bb_fft_out[i,j,k,:] = b_magnitude * b_fft_hat
                end
                
                # Assign the decomposed velocity components
                vv_fft_out[i,j,k,:] = v_fft_out 
            end
        end
    end

    # Perform inverse FFT and shift back the zero-frequency component
    iv_out = real(ifft(ifftshift(vv_fft_out[:,:,:,1])))
    jv_out = real(ifft(ifftshift(vv_fft_out[:,:,:,2])))
    kv_out = real(ifft(ifftshift(vv_fft_out[:,:,:,3])))

    ib_out = real(ifft(ifftshift(bb_fft_out[:,:,:,1])))
    jb_out = real(ifft(ifftshift(bb_fft_out[:,:,:,2])))
    kb_out = real(ifft(ifftshift(bb_fft_out[:,:,:,3])))

    return iv_out, jv_out, kv_out, ib_out, jb_out, kb_out
end

#***********************************************************************
end # module
