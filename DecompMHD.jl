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
export decomposition_A_m2, decomposition_F_m2, decomposition_S_m2

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

end # module
