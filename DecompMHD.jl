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

#***********************************************************************

"""
    function decomposition_A(
        iv::Cube, jv::Cube, kv::Cube, 
        ib::Cube, jb::Cube, kb::Cube, 
        v_A::Float64, c_s::Float64; 
        epsilon=1e-9::Float64, ρ0=1.0::Float64)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube}

Decompose the velocity and magnetic field cubes into Alfvén mode components.

# Arguments
- `iv`: Input velocity component in the i direction.
- `jv`: Input velocity component in the j direction.
- `kv`: Input velocity component in the k direction.
- `ib`: Input magnetic field component in the i direction.
- `jb`: Input magnetic field component in the j direction.
- `kb`: Input magnetic field component in the k direction.
- `v_A`: Alfvén velocity.
- `c_s`: Sound speed.
- `epsilon`: Small value to avoid division by zero (default: 1e-9).
- `ρ0`: Initial density (default: 1.0).

# Returns
- `iv_out, jv_out, kv_out`: The decomposed velocity components.
- `ib_out, jb_out, kb_out`: The decomposed magnetic field components.
"""
function decomposition_A(
    iv::Cube, jv::Cube, kv::Cube, 
    ib::Cube, jb::Cube, kb::Cube, 
    v_A::Float64, c_s::Float64; 
    epsilon=1e-9::Float64, ρ0=1.0::Float64)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube}
    
    # Get the dimensions of the input cubes
    nx, ny, nz = size(ib)
    
    # Calculate α = β * (γ / 2)
    α = c_s^2 / v_A^2

    # Perform FFT and shift the zero-frequency component to the center
    iv_fft = fftshift(fft(iv))
    jv_fft = fftshift(fft(jv))
    kv_fft = fftshift(fft(kv))

    # Initialize the Fourier output arrays
    vv_fft_out = zeros(typeof(1.0 + im), size(iv_fft)..., 3)
    bb_fft_out = zeros(typeof(1.0 + im), size(iv_fft)..., 3)

    # Calculate the mean magnetic field and unit vector
    B0 = [mean(ib), mean(jb), mean(kb)]
    Bhat = normalize(B0)

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
        if sum(kk .^ 2.0) > 0
            # Calculate the dot product of khat and Bhat (cosθ)
            cosθ = dot(khat, Bhat)

            # Alfven mode wave speed
            c_wave = v_A * cosθ

            # Induction equation denominator
            # ind_denom = c_wave * v_A * sqrt(ρ0)
            
            # Remove k_perp = 0 case
            if 1.0 - abs(cosθ) > epsilon
				#= 
                Get the Alfven mode basis vector
                ζ_Alf = k x B
                =#
                ζ_mode = normalize(cross(khat, Bhat))

                # Get projected Fourier velocity vector
                v_fft_out = dot(v_fft, ζ_mode) .* ζ_mode

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
        ib::Cube, jb::Cube, kb::Cube, 
        v_A::Float64, c_s::Float64; 
        epsilon=1e-9::Float64, ρ0=1.0::Float64)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube}

Decompose the velocity and magnetic field cubes into fast mode components.

# Arguments
- `iv`: Input velocity component in the i direction.
- `jv`: Input velocity component in the j direction.
- `kv`: Input velocity component in the k direction.
- `ib`: Input magnetic field component in the i direction.
- `jb`: Input magnetic field component in the j direction.
- `kb`: Input magnetic field component in the k direction.
- `v_A`: Alfvén velocity.
- `c_s`: Sound speed.
- `epsilon`: Small value to avoid division by zero (default: 1e-9).
- `ρ0`: Initial density (default: 1.0).

# Returns
- `iv_out, jv_out, kv_out`: The decomposed velocity components.
- `ib_out, jb_out, kb_out`: The decomposed magnetic field components.
"""
function decomposition_F(
    iv::Cube, jv::Cube, kv::Cube, 
    ib::Cube, jb::Cube, kb::Cube, 
    v_A::Float64, c_s::Float64; 
    epsilon=1e-9::Float64, ρ0=1.0::Float64)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube}
    
    # Get the dimensions of the input cubes
    nx, ny, nz = size(ib)
    
    # Calculate α = β * (γ / 2)
    α = c_s^2 / v_A^2
    
    # Factor for magnetosonic mode wave speeds
    ms_factor = v_A^2 + c_s^

    # Perform FFT and shift the zero-frequency component to the center
    iv_fft = fftshift(fft(iv))
    jv_fft = fftshift(fft(jv))
    kv_fft = fftshift(fft(kv))

    # Initialize the Fourier output arrays
    vv_fft_out = zeros(typeof(1.0 + im), size(iv_fft)..., 3)
    bb_fft_out = zeros(typeof(1.0 + im), size(iv_fft)..., 3)

    # Calculate the mean magnetic field and unit vector
    B0 = [mean(ib), mean(jb), mean(kb)]
    Bhat = normalize(B0)

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
        if sum(kk .^ 2.0) > 0
            # Calculate the dot product of khat and Bhat (cosθ)
            cosθ = dot(khat, Bhat)

            # Fast mode wave speed
            Δ2 = (ms_factor)^2 - (4.0 * v_A^2 * c_s^2 * cosθ^2)
            c_wave = sqrt(0.5 * (ms_factor + sqrt(Δ2)))

            # Induction equation denominator
            # ind_denom = c_wave * v_A * sqrt(ρ0)
            
            D = (1.0 + α)^2 - (4.0 * α * cosθ^2)
            prefactor_para = - 1.0 + α + sqrt(D)
            prefactor_perp = 1.0 + α + sqrt(D)

            # Remove k_perp = 0 case
            if 1.0 - abs(cosθ) > epsilon
				#= 
                Get the fast mode basis vector
                ζ_fast ∝ (- 1 + α + sqrt(D)) * k_para + 
                    (1 + α + sqrt(D)) * k_perp
                =#
                k_para = dot(kk, Bhat) .* Bhat
                k_perp = kk .- k_para
                ζ_mode = normalize((prefactor_para .* k_para) .+ (prefactor_perp .* k_perp))

                # Get projected Fourier velocity vector
                v_fft_out = dot(v_fft, ζ_mode) .* ζ_mode

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
        ib::Cube, jb::Cube, kb::Cube, 
        v_A::Float64, c_s::Float64; 
        epsilon=1e-9::Float64, ρ0=1.0::Float64)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube}

Decompose the velocity and magnetic field cubes into slow mode components.

# Arguments
- `iv`: Input velocity component in the i direction.
- `jv`: Input velocity component in the j direction.
- `kv`: Input velocity component in the k direction.
- `ib`: Input magnetic field component in the i direction.
- `jb`: Input magnetic field component in the j direction.
- `kb`: Input magnetic field component in the k direction.
- `v_A`: Alfvén velocity.
- `c_s`: Sound speed.
- `epsilon`: Small value to avoid division by zero (default: 1e-9).
- `ρ0`: Initial density (default: 1.0).

# Returns
- `iv_out, jv_out, kv_out`: The decomposed velocity components.
- `ib_out, jb_out, kb_out`: The decomposed magnetic field components.
"""
function decomposition_S(
    iv::Cube, jv::Cube, kv::Cube, 
    ib::Cube, jb::Cube, kb::Cube, 
    v_A::Float64, c_s::Float64; 
    epsilon=1e-9::Float64, ρ0=1.0::Float64)::Tuple{Cube,Cube,Cube,Cube,Cube,Cube}
    
    # Get the dimensions of the input cubes
    nx, ny, nz = size(ib)
    
    # Calculate α = β * (γ / 2)
    α = c_s^2 / v_A^2
    
    # Factor for magnetosonic mode wave speeds
    ms_factor = v_A^2 + c_s^2

    # Perform FFT and shift the zero-frequency component to the center
    iv_fft = fftshift(fft(iv))
    jv_fft = fftshift(fft(jv))
    kv_fft = fftshift(fft(kv))

    # Initialize the Fourier output arrays
    vv_fft_out = zeros(typeof(1.0 + im), size(iv_fft)..., 3)
    bb_fft_out = zeros(typeof(1.0 + im), size(iv_fft)..., 3)

    # Calculate the mean magnetic field and unit vector
    B0 = [mean(ib), mean(jb), mean(kb)]
    Bhat = normalize(B0)

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
        if sum(kk .^ 2.0) > 0
            # Calculate the dot product of khat and Bhat (cosθ)
            cosθ = dot(khat, Bhat)

            # Fast mode wave speed
            Δ2 = (ms_factor)^2 - (4 * v_A^2 * c_s^2 * cosθ^2)
            c_wave = sqrt(0.5 * (ms_factor - sqrt(Δ2)))

            # Induction equation denominator
            # ind_denom = c_wave * v_A * sqrt(ρ0)
            
            D = (1 + α)^2 - (4 * α * cosθ^2)
            prefactor_para = - 1 + α - sqrt(D)
            prefactor_perp = 1 + α - sqrt(D)

            # Remove k_perp = 0 case
            if 1.0 - abs(cosθ) > epsilon
				#= 
                Get the slow mode basis vector
                ζ_fast ∝ (- 1 + α - sqrt(D)) * k_para + 
                    (1 + α - sqrt(D)) * k_perp
                =#
                k_para = dot(kk, Bhat) .* Bhat
                k_perp = kk .- k_para
                ζ_mode = normalize((prefactor_para .* k_para) .+ (prefactor_perp .* k_perp))

                # Get projected Fourier velocity vector
                v_fft_out = dot(v_fft, ζ_mode) .* ζ_mode

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


end #top
