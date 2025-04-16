#***********************************************************************
module DecompMHD
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

export decomposition_A, decomposition_F, decomposition_S
export decomposition_AC_B, decomposition_all_V

#***********************************************************************

"""
    function decomposition_A(
        iv::Cube, jv::Cube, kv::Cube, 
        B0::Vect, β::Float64;
        ε::Float64=(0.5/size(iv)[1]^2), ρ0::Float64=1.0)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube}

Decompose the velocity and magnetic field cubes into Alfvén mode components.

# Arguments
- `iv`: Input velocity component in the i direction.
- `jv`: Input velocity component in the j direction.
- `kv`: Input velocity component in the k direction.
- `B0`: Mean magnetic field vector.
- `β` : Plasma β (2 * c_s^2 / v_A^2).
- `ε` : Small value to avoid division by zero (default: 0.5 / box_size^2).
- `ρ0`: Initial density (default: 1.0).

# Returns
- `iv_out, jv_out, kv_out`: The decomposed velocity components.
- `ib_out, jb_out, kb_out`: The decomposed magnetic field components.
"""
function decomposition_A(
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
    vv_fft_out = zeros(ComplexF64, size(iv_fft)..., 3)
    bb_fft_out = zeros(ComplexF64, size(iv_fft)..., 3)

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

                if abs(c_wave) > 0
                    # Get B-field Fourier components
                    b_fft_out = cross(khat, cross(v_fft_out, B0)) ./ c_wave
                else
                    # Remove non-propagating modes
                    b_fft_out = [0., 0., 0.]
                end
                
                # Assign the decomposed velocity components
                vv_fft_out[i,j,k,:] = v_fft_out
                
                # Assign the decomposed magnetic field components
                bb_fft_out[i,j,k,:] = b_fft_out

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

"""
    function decomposition_F(
        iv::Cube, jv::Cube, kv::Cube, 
        B0::Vect, β::Float64; 
        ε::Float64=(0.5/size(iv)[1]^2), ρ0::Float64=1.0)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube}

Decompose the velocity and magnetic field cubes into fast mode components.

# Arguments
- `iv`: Input velocity component in the i direction.
- `jv`: Input velocity component in the j direction.
- `kv`: Input velocity component in the k direction.
- `B0`: Mean magnetic field vector.
- `β` : Plasma β (2 * c_s^2 / v_A^2).
- `ε` : Small value to avoid division by zero (default: 0.5 / box_size^2).
- `ρ0`: Initial density (default: 1.0).

# Returns
- `iv_out, jv_out, kv_out`: The decomposed velocity components.
- `ib_out, jb_out, kb_out`: The decomposed magnetic field components.
"""
function decomposition_F(
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
    vv_fft_out = zeros(ComplexF64, size(iv_fft)..., 3)
    bb_fft_out = zeros(ComplexF64, size(iv_fft)..., 3)

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

            # Induction equation denominator
            # ind_denom = c_wave * v_A * sqrt(ρ0)
            
            # Factors from CL03
            D = (1.0 + α)^2 - (4.0 * α * cosθ^2)
            prefactor_para = - 1.0 + α + sqrt(D)
            prefactor_perp = 1.0 + α + sqrt(D)

            # Fast mode wave speed
            c_wave = v_A * sqrt(0.5 * prefactor_perp)

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

                if abs(c_wave) > 0
                    # Get B-field Fourier components
                    b_fft_out = cross(khat, cross(v_fft_out, B0)) ./ c_wave
                else
                    # Remove non-propagating modes
                    b_fft_out = [0., 0., 0.]
                end
                
                # Assign the decomposed velocity components
                vv_fft_out[i,j,k,:] = v_fft_out

                # Assign the decomposed magnetic field components
                bb_fft_out[i,j,k,:] = b_fft_out

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

"""
    function decomposition_S(
        iv::Cube, jv::Cube, kv::Cube, 
        B0::Vect, β::Float64;
        ε::Float64=(0.5/size(iv)[1]^2), ρ0::Float64=1.0)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube}

Decompose the velocity and magnetic field cubes into slow mode components.

# Arguments
- `iv`: Input velocity component in the i direction.
- `jv`: Input velocity component in the j direction.
- `kv`: Input velocity component in the k direction.
- `B0`: Mean magnetic field vector.
- `β` : Plasma β (2 * c_s^2 / v_A^2).
- `ε` : Small value to avoid division by zero (default: 0.5 / box_size^2).
- `ρ0`: Initial density (default: 1.0).

# Returns
- `iv_out, jv_out, kv_out`: The decomposed velocity components.
- `ib_out, jb_out, kb_out`: The decomposed magnetic field components.
"""
function decomposition_S(
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
    vv_fft_out = zeros(ComplexF64, size(iv_fft)..., 3)
    bb_fft_out = zeros(ComplexF64, size(iv_fft)..., 3)

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

            # Induction equation denominator
            # ind_denom = c_wave * v_A * sqrt(ρ0)
            
            # Factors from CL03
            D = (1.0 + α)^2 - (4.0 * α * cosθ^2)
            prefactor_para = - 1.0 + α - sqrt(D)
            prefactor_perp = 1.0 + α - sqrt(D)

            # Slow mode wave speed
            c_wave = v_A * sqrt(0.5 * prefactor_perp)

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

                if abs(c_wave) > 0
                    # Get B-field Fourier components
                    b_fft_out = cross(khat, cross(v_fft_out, B0)) ./ c_wave
                    # b_fft_out = norm(B0) .* norm(v_fft_out) .* cross(normalize(k_para), ξ_mode) ./ c_wave
                else
                    # Remove non-propagating modes
                    b_fft_out = [0., 0., 0.]
                end
                
                # Assign the decomposed velocity components
                vv_fft_out[i,j,k,:] = v_fft_out
                
                # Assign the decomposed magnetic field components
                bb_fft_out[i,j,k,:] = b_fft_out

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


function decomposition_AC_B(
    ib::Cube, jb::Cube, kb::Cube, 
    B0::Vect, β::Float64; 
    ε::Float64=(0.5/size(ib)[1]^2), ρ0::Float64=1.0)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube}
    
    # Get the dimensions of the input cubes
    nx, ny, nz = size(ib)

    # Calculate the mean magnetic field unit vector
    Bhat = normalize(B0)
    
    # Calculate v_A and c_s
    v_A = norm(B0) / sqrt(ρ0)
    c_s = sqrt(β * v_A^2 / 2)
    
    # Calculate α = β * (γ / 2)
    α = β / 2.0

    # Perform FFT and shift the zero-frequency component to the center
    ib_fft = fftshift(fft(ib))
    jb_fft = fftshift(fft(jb))
    kb_fft = fftshift(fft(kb))

    # Initialize the Fourier output arrays
    bb_fft_out_a = zeros(ComplexF64, size(ib_fft)..., 3)
    bb_fft_out_c = zeros(ComplexF64, size(ib_fft)..., 3)

    for k in 1:nz, j in 1:ny, i in 1:nx

        # Assign vecocity Fourier vector at (i,j,k)
        b_fft = [ib_fft[i, j, k], jb_fft[i, j, k], kb_fft[i, j, k]]

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

            # Remove k_perp = 0 case
            if 1.0 - abs(cosθ) > ε
				#= 
                Get the Alfven mode dB unit vector
                ξ_alf = (k x B)
                =#
                ξ_alfv = normalize(cross(khat, Bhat))
                
                #= 
                Get the MS mode dB unit vector
                ξ_MS = k x (k x B)
                =#
                ξ_comp = normalize(cross(khat, cross(khat, Bhat)))

                # Get projected Fourier B-field vector
                bb_fft_out_a[i,j,k,:] = dot(b_fft, ξ_alfv) .* ξ_alfv
                bb_fft_out_c[i,j,k,:] = dot(b_fft, ξ_comp) .* ξ_comp
            end
        end
    end

    # Perform inverse FFT and shift back the zero-frequency component
    ib_out_a = real(ifft(ifftshift(bb_fft_out_a[:,:,:,1])))
    jb_out_a = real(ifft(ifftshift(bb_fft_out_a[:,:,:,2])))
    kb_out_a = real(ifft(ifftshift(bb_fft_out_a[:,:,:,3])))

    ib_out_c = real(ifft(ifftshift(bb_fft_out_c[:,:,:,1])))
    jb_out_c = real(ifft(ifftshift(bb_fft_out_c[:,:,:,2])))
    kb_out_c = real(ifft(ifftshift(bb_fft_out_c[:,:,:,3])))

    return ib_out_a, jb_out_a, kb_out_a, ib_out_c, jb_out_c, kb_out_c
end



function decomposition_SF_B(
    ib::Cube, jb::Cube, kb::Cube, 
    B0::Vect, β::Float64; 
    ε::Float64=(0.5/size(ib)[1]^2), ρ0::Float64=1.0)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube}
    
    # Get the dimensions of the input cubes
    nx, ny, nz = size(ib)

    # Calculate the mean magnetic field unit vector
    Bhat = normalize(B0)
    
    # Calculate v_A and c_s
    v_A = norm(B0) / sqrt(ρ0)
    c_s = sqrt(β * v_A^2 / 2)
    
    # Calculate α = β * (γ / 2)
    α = β / 2.0

    # Perform FFT and shift the zero-frequency component to the center
    ib_fft = fftshift(fft(ib))
    jb_fft = fftshift(fft(jb))
    kb_fft = fftshift(fft(kb))

    # Initialize the Fourier output arrays
    bb_fft_out_f = zeros(ComplexF64, size(ib_fft)..., 3)
    bb_fft_out_s = zeros(ComplexF64, size(ib_fft)..., 3)

    for k in 1:nz, j in 1:ny, i in 1:nx

        # Calculate the unit wave vector
        kx = i - div(nx, 2) - 1
        ky = j - div(ny, 2) - 1
        kz = k - div(nz, 2) - 1
        kk = [kx, ky, kz]
        khat = normalize(kk)

        # Assign vecocity Fourier vector at (i,j,k)
        b_fft = [ib_fft[i, j, k], jb_fft[i, j, k], kb_fft[i, j, k]]

        # Avoid zero wave vectors
        if !iszero(norm(kk))
            # Calculate the dot product of khat and Bhat (cosθ)
            cosθ = dot(khat, Bhat)
            
            D = (1.0 + α)^2 - (4.0 * α * cosθ^2)
            prefactor_para_f = - 1.0 + α + sqrt(D)
            prefactor_perp_f = 1.0 + α + sqrt(D)

            prefactor_para_s = - 1.0 + α - sqrt(D)
            prefactor_perp_s = 1.0 + α - sqrt(D)

            # Slow/Fast mode wave speed
            c_fast = v_A * sqrt(0.5 * prefactor_perp_f)
            c_slow = v_A * sqrt(0.5 * prefactor_perp_s)
            
            # Remove k_perp = 0 case
            if 1.0 - abs(cosθ) > ε

                k_para = dot(kk, Bhat) .* Bhat
                k_perp = kk .- k_para
                
                ξ_fast = normalize((prefactor_para_f .* k_para) .+ (prefactor_perp_f .* k_perp))
                ξ_slow = normalize((prefactor_para_s .* k_para) .+ (prefactor_perp_s .* k_perp))

                n_fast = abs(c_fast) > ε ? (dot(ξ_fast, k_perp) ./ c_fast) : 0.0
                n_slow = abs(c_slow) > ε ? (dot(ξ_slow, k_perp) ./ c_slow) : 0.0

                if abs(cosθ) > ε
                    norm_fast = n_fast ./ (n_fast + n_slow)
                    norm_slow = n_slow ./ (n_fast + n_slow)
                else
                    norm_fast = 1.0
                    norm_slow = 0.0
                end
                
                ξ_comp = normalize(cross(khat, cross(khat, Bhat)))

                bb_fft_out_f[i,j,k,:] = dot(b_fft, ξ_comp) * ξ_comp * norm_fast
                bb_fft_out_s[i,j,k,:] = dot(b_fft, ξ_comp) * ξ_comp * norm_slow
            end
        end
    end

    ib_out_f = real(ifft(ifftshift(bb_fft_out_f[:,:,:,1])))
    jb_out_f = real(ifft(ifftshift(bb_fft_out_f[:,:,:,2])))
    kb_out_f = real(ifft(ifftshift(bb_fft_out_f[:,:,:,3])))

    ib_out_s = real(ifft(ifftshift(bb_fft_out_s[:,:,:,1])))
    jb_out_s = real(ifft(ifftshift(bb_fft_out_s[:,:,:,2])))
    kb_out_s = real(ifft(ifftshift(bb_fft_out_s[:,:,:,3])))

    return ib_out_f, jb_out_f, kb_out_f, ib_out_s, jb_out_s, kb_out_s

end

function decomposition_all_V(
    iv::Cube, jv::Cube, kv::Cube, 
    B0::Vect, β::Float64;
    ε::Float64=(0.5/size(iv)[1]^2), ρ0::Float64=1.0)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube,Cube,Cube,Cube}
    
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
    vv_fft_out_a = zeros(ComplexF64, size(iv_fft)..., 3)
    vv_fft_out_f = zeros(ComplexF64, size(iv_fft)..., 3)
    vv_fft_out_s = zeros(ComplexF64, size(iv_fft)..., 3)

    for k in 1:nz, j in 1:ny, i in 1:nx
        
        # Calculate the unit wave vector
        kx = i - div(nx, 2) - 1
        ky = j - div(ny, 2) - 1
        kz = k - div(nz, 2) - 1
        kk = [kx, ky, kz]
        khat = normalize(kk)

        # Assign vecocity Fourier vector at (i,j,k)
        v_fft = [iv_fft[i, j, k], jv_fft[i, j, k], kv_fft[i, j, k]]
 
        # Avoid zero wave vectors
        if !iszero(norm(kk))
            # Calculate the dot product of khat and Bhat (cosθ)
            cosθ = dot(khat, Bhat)
            
            # Factors from CL03
            D = (1.0 + α)^2 - (4.0 * α * cosθ^2)

            prefactor_para_f = - 1.0 + α + sqrt(D)
            prefactor_perp_f = 1.0 + α + sqrt(D)

            prefactor_para_s = - 1.0 + α - sqrt(D)
            prefactor_perp_s = 1.0 + α - sqrt(D)

            # Mode wave speeds
            c_alfv = v_A * cosθ
            c_fast = v_A * sqrt(0.5 * prefactor_perp_f)
            c_slow = v_A * sqrt(0.5 * prefactor_perp_s)

            # Remove k_perp = 0 case    
            if 1.0 - abs(cosθ) > ε
				k_para = dot(kk, Bhat) .* Bhat
                k_perp = kk .- k_para
                
                #= 
                Get the Alfven mode basis vector
                ξ_Alf = k x B
                =#
                ξ_alfv = normalize(cross(khat, Bhat))
                
                #= 
                Get the fast mode basis vector
                ξ_fast ∝ (- 1 + α + sqrt(D)) * k_para + 
                    (1 + α + sqrt(D)) * k_perp
                =#
                ξ_fast = normalize((prefactor_para_f .* k_para) .+ (prefactor_perp_f .* k_perp))

				#= 
                Get the slow mode basis vector
                ξ_fast ∝ (- 1 + α - sqrt(D)) * k_para + 
                    (1 + α - sqrt(D)) * k_perp
                =#
                ξ_slow = normalize((prefactor_para_s .* k_para) .+ (prefactor_perp_s .* k_perp))

                # Get projected Fourier velocity vectors
                vv_fft_out_a[i,j,k,:] = dot(v_fft, ξ_alfv) .* ξ_alfv
                vv_fft_out_f[i,j,k,:] = dot(v_fft, ξ_fast) .* ξ_fast
                vv_fft_out_s[i,j,k,:] = dot(v_fft, ξ_slow) .* ξ_slow

            end
        end
    end

    # Perform inverse FFT and shift back the zero-frequency component
    iv_out_a = real(ifft(ifftshift(vv_fft_out_a[:,:,:,1])))
    jv_out_a = real(ifft(ifftshift(vv_fft_out_a[:,:,:,2])))
    kv_out_a = real(ifft(ifftshift(vv_fft_out_a[:,:,:,3])))

    iv_out_f = real(ifft(ifftshift(vv_fft_out_f[:,:,:,1])))
    jv_out_f = real(ifft(ifftshift(vv_fft_out_f[:,:,:,2])))
    kv_out_f = real(ifft(ifftshift(vv_fft_out_f[:,:,:,3])))

    iv_out_s = real(ifft(ifftshift(vv_fft_out_s[:,:,:,1])))
    jv_out_s = real(ifft(ifftshift(vv_fft_out_s[:,:,:,2])))
    kv_out_s = real(ifft(ifftshift(vv_fft_out_s[:,:,:,3])))

    return iv_out_a, jv_out_a, kv_out_a, iv_out_f, jv_out_f, kv_out_f, iv_out_s, jv_out_s, kv_out_s
end


end # module
