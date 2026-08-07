#***********************************************************************
module TurbAnalysis
#***********************************************************************

# MATH PACKAGES
using Statistics
using FFTW, Random
using LinearAlgebra
using Images

# CUSTOM PACKAGES
using DataStruct
using BaseIO
using ArrayFunctions
using DecompMHD

#***********************************************************************

export energy_spectrum_all
export oned_energy_spectrum, oned_energy_spectrum_para, oned_energy_spectrum_perp
export twod_energy_spectrum, oned_energy_spectrum_matx, oned_psd_matx
export fourier_filter_2D, fourier_filter_3D
export get_ma_dv, get_ma_db, get_ma2d_dv, get_ma2d_db, get_ms, entot, entot_weighted
export get_enfrac_ac, get_enfrac_asf, get_enfrac_asf_total, enfrac_vs_time
export sf2d, pairwise_annular_average

#***********************************************************************

function energy_spectrum_all(u::Cube, v::Cube, w::Cube)::Tuple{Vect, Vect, Vect, Matx}
    
    # Perform FFT on each component of the field
    u_fft = fft(u)
    v_fft = fft(v)
    w_fft = fft(w)

    # Compute the power spectrum
    power_spectrum = fftshift(abs2.(u_fft) + abs2.(v_fft) + abs2.(w_fft))
    power_spectrum_sum_para = sum(power_spectrum, dims=(1, 2))
    power_spectrum_sum_perp = sum(power_spectrum, dims=3)
    
    # Average over spherical shells
    nx, ny, nz = size(u)
    kx = collect(0:nx-1) .- (nx / 2)
    ky = collect(0:ny-1) .- (ny / 2)
    kz = collect(0:nz-1) .- (nz / 2)
    kx_g, ky_g, kz_g = meshgrid_3D(kx, ky, kz)
    kx_g2, ky_g2 = meshgrid_2D(kx, ky)
    
    # 1D spectrum
    ksq = kx_g.^2 .+ ky_g.^2 .+ kz_g.^2
    kbins1d = round.(Int, sqrt.(ksq))
    Pk1d = zeros(maximum(kbins1d) + 1)

    # 2D spectrum
    ksq = kx_g2.^2 .+ ky_g2.^2
    kbins2d = round.(Int, sqrt.(ksq))
    Pk2d = zeros(nz, maximum(kbins2d) + 1)

    # 1D parallel spectrum
    ksq = kz.^2
    kbinspar = round.(Int, sqrt.(ksq))
    Pk1d_para = zeros(maximum(kbinspar) + 1)
    
    for k in 1:nz
        Pk1d_para[kbinspar[k] + 1] += power_spectrum_sum_para[k]
        for j in 1:ny, i in 1:nx 
        Pk1d[kbins1d[i,j,k] + 1] += power_spectrum[i,j,k]
        Pk2d[k, kbins2d[i,j] + 1] += power_spectrum[i,j,k]
        end
    end 

    Pk1d_perp = sum(Pk2d, dims=1)[1,:]

    return Pk1d, Pk1d_para, Pk1d_perp, Pk2d[div(nz,2)+2:end, 2:div(nx,2)]
end


function oned_energy_spectrum(u::Cube)::Vect
    
    # Perform FFT on each component of the field
    u_fft = fft(u)

    # Compute the power spectrum
    power_spectrum = fftshift(abs2.(u_fft))
    
    # Average over spherical shells
    nx, ny, nz = size(u)
    kx = collect(0:nx-1) .- (nx / 2)
    ky = collect(0:ny-1) .- (ny / 2)
    kz = collect(0:nz-1) .- (nz / 2)
    kx_g, ky_g, kz_g = meshgrid_3D(kx, ky, kz)
    
    ksq = kx_g.^2 .+ ky_g.^2 .+ kz_g.^2
    kbins = round.(Int, sqrt.(ksq))

    Pk = zeros(maximum(kbins) + 1)
    [Pk[kbins[i,j,k] + 1] += power_spectrum[i,j,k] for k in 1:nz, j in 1:ny, i in 1:nx]

    return Pk
end


function oned_energy_spectrum(u::Cube, v::Cube)
    
    # Perform FFT on each component of the field
    u_fft = fft(u)
    v_fft = fft(v)

    # Compute the power spectrum
    power_spectrum = fftshift(abs2.(u_fft) + abs2.(v_fft))::Vect
    
    # Average over spherical shells
    nx, ny, nz = size(u)
    kx = collect(0:nx-1) .- (nx / 2)
    ky = collect(0:ny-1) .- (ny / 2)
    kz = collect(0:nz-1) .- (nz / 2)
    kx_g, ky_g, kz_g = meshgrid_3D(kx, ky, kz)
    
    ksq = kx_g.^2 .+ ky_g.^2 .+ kz_g.^2
    kbins = round.(Int, sqrt.(ksq))

    Pk = zeros(maximum(kbins) + 1)
    [Pk[kbins[i,j,k] + 1] += power_spectrum[i,j,k] for k in 1:nz, j in 1:ny, i in 1:nx]

    return Pk
end

function oned_energy_spectrum(u::Cube, v::Cube, w::Cube)::Vect
    
    # Perform FFT on each component of the field
    u_fft = fft(u)
    v_fft = fft(v)
    w_fft = fft(w)

    # Compute the power spectrum
    power_spectrum = fftshift(abs2.(u_fft) + abs2.(v_fft) + abs2.(w_fft))
    
    # Average over spherical shells
    nx, ny, nz = size(u)
    kx = collect(0:nx-1) .- (nx / 2)
    ky = collect(0:ny-1) .- (ny / 2)
    kz = collect(0:nz-1) .- (nz / 2)
    kx_g, ky_g, kz_g = meshgrid_3D(kx, ky, kz)
    
    ksq = kx_g.^2 .+ ky_g.^2 .+ kz_g.^2
    kbins = round.(Int, sqrt.(ksq))

    Pk = zeros(maximum(kbins) + 1)
    [Pk[kbins[i,j,k] + 1] += power_spectrum[i,j,k] for k in 1:nz, j in 1:ny, i in 1:nx]
    
    return Pk
end


function oned_energy_spectrum_para(u::Cube, v::Cube, w::Cube)::Vect
    
    # Perform FFT on each component of the field
    u_fft = fft(u)
    v_fft = fft(v)
    w_fft = fft(w)

    # Compute the power spectrum
    power_spectrum = fftshift(abs2.(u_fft) + abs2.(v_fft) + abs2.(w_fft))
    power_spectrum_sum = sum(power_spectrum, dims=(1, 2))
    
    # Average over spherical shells
    nx, ny, nz = size(u)
    kz = collect(0:nz-1) .- (nz / 2)

    ksq = kz.^2
    kbins = round.(Int, sqrt.(ksq))

    Pk = zeros(maximum(kbins) + 1)
    [Pk[kbins[k] + 1] += power_spectrum_sum[k] for k in 1:nz]

    return Pk
end



function oned_energy_spectrum_perp(u::Cube, v::Cube, w::Cube)::Vect
    
    # Perform FFT on each component of the field
    u_fft = fft(u)
    v_fft = fft(v)
    w_fft = fft(w)

    # Compute the power spectrum
    power_spectrum = fftshift(abs2.(u_fft) + abs2.(v_fft) + abs2.(w_fft))
    power_spectrum_sum = sum(power_spectrum, dims=3)
    
    # Average over spherical shells
    nx, ny, nz = size(u)
    kx = collect(0:nx-1) .- (nx / 2)
    ky = collect(0:ny-1) .- (ny / 2)
    kx_g, ky_g = meshgrid_2D(kx, ky)
    
    ksq = kx_g.^2 .+ ky_g.^2
    kbins = round.(Int, sqrt.(ksq))

    Pk = zeros(maximum(kbins) + 1)
    [Pk[kbins[i,j] + 1] += power_spectrum_sum[i,j] for j in 1:ny, i in 1:nx]

    return Pk
end

function twod_energy_spectrum(u::Cube, v::Cube, w::Cube)::Matx
    
    # Perform FFT on each component of the field
    u_fft = fft(u)
    v_fft = fft(v)
    w_fft = fft(w)

    # Compute the power spectrum
    power_spectrum = fftshift(abs2.(u_fft) + abs2.(v_fft) + abs2.(w_fft))
    
    # Average over spherical shells
    nx, ny, nz = size(u)
    kx = collect(0:nx-1) .- (nx / 2)
    ky = collect(0:ny-1) .- (ny / 2)
    kx_g, ky_g = meshgrid_2D(kx, ky)
    
    ksq = kx_g.^2 .+ ky_g.^2
    kbins = round.(Int, sqrt.(ksq))

    Pk = zeros(nz, maximum(kbins) + 1)
    [Pk[k, kbins[i,j] + 1] += power_spectrum[i,j,k] for k in 1:nz, j in 1:ny, i in 1:nx]

    return Pk[div(nz,2)+2:end, 2:div(nx,2)]
end

function oned_energy_spectrum_matx(map::Matx)::Vect
    
    # Perform FFT on each component of the field
    map_fft = fft(map)

    # Compute the power spectrum
    power_spectrum = fftshift(abs2.(map_fft))
    
    # Average over spherical shells
    nx, ny = size(map)
    kx = collect(0:nx-1) .- (nx / 2)
    ky = collect(0:ny-1) .- (ny / 2)
    kx_g, ky_g = meshgrid_2D(kx, ky)
    
    ksq = kx_g.^2 .+ ky_g.^2
    kbins = round.(Int, sqrt.(ksq))

    Pk = zeros(maximum(kbins) + 1)
    [Pk[kbins[i,j] + 1] += power_spectrum[i,j] for j in 1:ny, i in 1:nx]
    
    return Pk
end


function oned_psd_matx(map::Matx)::Vect
    
    # Perform FFT on each component of the field
    map_fft = fft(map)

    # Compute the power spectrum
    power_spectrum = fftshift(abs2.(map_fft))
    
    # Average over spherical shells
    nx, ny = size(map)
    kx = collect(0:nx-1) .- (nx / 2)
    ky = collect(0:ny-1) .- (ny / 2)
    kx_g, ky_g = meshgrid_2D(kx, ky)
    ksq = kx_g.^2 .+ ky_g.^2
    kbins = round.(Int, sqrt.(ksq))

    n_bins = maximum(kbins) + 1
    Pk = zeros(n_bins)
    counts = zeros(Int, n_bins)

    # Bin power and count number of modes in each shell
    for j in 1:ny, i in 1:nx
        k = kbins[i, j] + 1
        Pk[k] += power_spectrum[i, j]
        counts[k] += 1
    end

    PSD = Pk ./ counts

    return PSD
end


function fourier_filter_2D(cdata::Matx, k_cutoff::Float64; pass::String="high", keep_mean::Bool=false)::Matx

    # Get the dimensions of the matrix
    nx, ny = size(cdata)

    # Compute the Fourier transform of the matrix
    cdata_ft = fftshift(fft(cdata))

    # Initialize centered wavevector grid
    kx = collect(0:nx-1) .- (nx / 2)
    ky = collect(0:ny-1) .- (ny / 2)
    
    # Create a 3D grid of wavevector magnitudes
    kx_g, ky_g, = meshgrid_2D(kx, ky)
    kbins = sqrt.(kx_g.^2 .+ ky_g.^2)

    # Create the filter mask
    if pass == "high"
        filter_mask = kbins .>= k_cutoff
        # Keep mean component (k = 0) if requested
        if keep_mean filter_mask[kbins .== 0] .= true end
    elseif pass == "low"
        filter_mask = kbins .<= k_cutoff
    else
        error("Invalid filter pass type. Choose 'high' or 'low'")
    end
    
    # Apply to the Fourier data
    cdata_ft_filtered = cdata_ft .* filter_mask

    # Inverse Fourier transform
    cdata_filtered = real(ifft(ifftshift(cdata_ft_filtered)))

    return cdata_filtered
end

function fourier_filter_3D(cdata::Cube, k_cutoff::Float64; pass::String="high", keep_mean::Bool=false)::Cube

    # Get the dimensions of the cube
    nx, ny, nz = size(cdata)

    # Compute the Fourier transform of the cube
    cdata_ft = fftshift(fft(cdata))

    # Initialize centered wavevector grid
    kx = collect(0:nx-1) .- (nx / 2)
    ky = collect(0:ny-1) .- (ny / 2)
    kz = collect(0:nz-1) .- (nz / 2)
    
    # Create a 3D grid of wavevector magnitudes
    kx_g, ky_g, kz_g = meshgrid_3D(kx, ky, kz)
    kbins = sqrt.(kx_g.^2 .+ ky_g.^2 .+ kz_g.^2)

    # Create the filter mask
    if pass == "high"
        filter_mask = kbins .>= k_cutoff
        # Keep mean component (k = 0) if requested
        if keep_mean filter_mask[kbins .== 0] .= true end
    elseif pass == "low"
        filter_mask = kbins .<= k_cutoff
    else
        error("Invalid filter pass type. Choose 'high' or 'low'")
    end
    
    # Apply to the Fourier data
    cdata_ft_filtered = cdata_ft .* filter_mask

    # Inverse Fourier transform
    cdata_filtered = real(ifft(ifftshift(cdata_ft_filtered)))

    return cdata_filtered
end


function get_ma_dv(bx::Cube, by::Cube, bz::Cube, vx::Cube, vy::Cube, vz::Cube)::Float64
    bxmean, bymean, bzmean = mean(bx), mean(by), mean(bz)
    b0 = sqrt(bxmean^2 + bymean^2 + bzmean^2)
    dvx, dvy, dvz = vx .- mean(vx), vy .- mean(vy), vz .- mean(vz)
    dvrms = sqrt(mean(dvx.^2 + dvy.^2 + dvz.^2))
    return dvrms / b0
end


function get_ma_db(bx::Cube, by::Cube, bz::Cube)::Float64
    bxmean, bymean, bzmean = mean(bx), mean(by), mean(bz)
    b0 = sqrt(bxmean^2 + bymean^2 + bzmean^2)
    dbx, dby, dbz = bx .- mean(bx), by .- mean(by), bz .- mean(bz)
    dbrms = sqrt(mean(dbx.^2 + dby.^2 + dbz.^2))
    return dbrms / b0
end



function get_ma2d_dv(bx::Cube, by::Cube, vx::Cube, vy::Cube)::Float64
    bxmean, bymean = mean(bx), mean(by)
    b0 = sqrt(bxmean^2 + bymean^2)
    dvx, dvy = vx .- mean(vx), vy .- mean(vy)
    dvrms = sqrt(mean(dvx.^2 + dvy.^2))
    return dvrms / b0
end


function get_ma2d_db(bx::Cube, by::Cube)::Float64
    bxmean, bymean = mean(bx), mean(by)
    b0 = sqrt(bxmean^2 + bymean^2)
    dbx, dby = bx .- mean(bx), by .- mean(by)
    dbrms = sqrt(mean(dbx.^2 + dby.^2))
    return dbrms / b0
end


function get_ms(cs::Float64, vx::Cube, vy::Cube, vz::Cube)::Float64
    dvx, dvy, dvz = vx .- mean(vx), vy .- mean(vy), vz .- mean(vz)
    dvrms = sqrt(mean(dvx.^2 + dvy.^2 + dvz.^2))
    return dvrms / cs
end

function entot(bx::Cube, by::Cube, bz::Cube)::Float64
    return sum(bx.^2.0 .+ by.^2.0 .+ bz.^2.0)
end

function entot_weighted(bx::Cube, by::Cube, bz::Cube, dens::Cube)::Float64
    return sum((bx.^2.0 .+ by.^2.0 .+ bz.^2.0) .* dens)
end

function get_enfrac_ac(bx_A::Cube, by_A::Cube, bz_A::Cube, bx_C::Cube, by_C::Cube, bz_C::Cube)::Tuple{Float64, Float64}
    entot_A = entot(bx_A, by_A, bz_A)
    entot_C = entot(bx_C, by_C, bz_C)
    total_entot = entot_A + entot_C
    perb_A = 100 * (entot_A / total_entot)
    perb_C = 100 * (entot_C / total_entot)
    return perb_A, perb_C
end


function get_enfrac_asf(bx_A::Cube, by_A::Cube, bz_A::Cube, bx_F::Cube, by_F::Cube, bz_F::Cube, bx_S::Cube, by_S::Cube, bz_S::Cube)::Tuple{Float64, Float64, Float64}
    entot_A = entot(bx_A, by_A, bz_A)
    entot_F = entot(bx_F, by_F, bz_F)
    entot_S = entot(bx_S, by_S, bz_S)
    total_entot = entot_A + entot_F + entot_S
    perb_A = 100 * (entot_A / total_entot)
    perb_F = 100 * (entot_F / total_entot)
    perb_S = 100 * (entot_S / total_entot)
    return perb_A, perb_F, perb_S
end


function get_enfrac_asf_total(bx_A::Cube, by_A::Cube, bz_A::Cube, bx_F::Cube, by_F::Cube, bz_F::Cube, 
    bx_S::Cube, by_S::Cube, bz_S::Cube, vx_A::Cube, vy_A::Cube, vz_A::Cube, 
    vx_F::Cube, vy_F::Cube, vz_F::Cube, vx_S::Cube, vy_S::Cube, vz_S::Cube
    )::Tuple{Float64, Float64, Float64, Float64, Float64, Float64}
    
    en_tot = entot(bx_A, by_A, bz_A) + entot(bx_F, by_F, bz_F) + entot(bx_S, by_S, bz_S) + entot(vx_A, vy_A, vz_A) + entot(vx_F, vy_F, vz_F) + entot(vx_S, vy_S, vz_S)
        
    perb_bA = 100 * ( entot(bx_A, by_A, bz_A) / en_tot )
    perb_bF = 100 * ( entot(bx_F, by_F, bz_F) / en_tot )
    perb_bS = 100 * ( entot(bx_S, by_S, bz_S) / en_tot )
   
    perb_vA = 100 * ( entot(vx_A, vy_A, vz_A) / en_tot )  
    perb_vF = 100 * ( entot(vx_F, vy_F, vz_F) / en_tot )  
    perb_vS = 100 * ( entot(vx_S, vy_S, vz_S) / en_tot )
    
    return perb_bA, perb_bF, perb_bS, perb_vA, perb_vF, perb_vS
end

function enfrac_vs_time(data_vec::Vector{String}, nres::Integer; resize_factor=4::Integer, βin=1.0::Float64)::Array{Float64,3}
    
    n_resize = div(nres, resize_factor)
    me_ke_AFS = zeros(Float64, length(data_vec), 2, 3)

    println("Initiated \n")
    flush(stdout)
    
    for (it,data_path) in enumerate(data_vec)

        dens, vx_in, vy_in, vz_in, bx_in, by_in, bz_in = read_data_athdf(data_path, nres)

        println(data_path)
        flush(stdout)
        
        rho_0 = mean(dens)

        bx = imresize(bx_in, (n_resize, n_resize, n_resize))
        by = imresize(by_in, (n_resize, n_resize, n_resize))
        bz = imresize(bz_in, (n_resize, n_resize, n_resize))

        vx = imresize(vx_in, (n_resize, n_resize, n_resize))
        vy = imresize(vy_in, (n_resize, n_resize, n_resize))
        vz = imresize(vz_in, (n_resize, n_resize, n_resize))

        B0 = [mean(bx), mean(by), mean(bz)]

        vx_A, vy_A, vz_A, bx_A, by_A, bz_A = decomposition_A(vx, vy, vz, B0, βin, ρ0=rho_0)
        vx_F, vy_F, vz_F, bx_F, by_F, bz_F = decomposition_F(vx, vy, vz, B0, βin, ρ0=rho_0)
        vx_S, vy_S, vz_S, bx_S, by_S, bz_S = decomposition_S(vx, vy, vz, B0, βin, ρ0=rho_0)

        me_ke_AFS[it,1,1] = entot(bx_A, by_A, bz_A)
        me_ke_AFS[it,1,2] = entot(bx_F, by_F, bz_F)
        me_ke_AFS[it,1,3] = entot(bx_S, by_S, bz_S)
        me_ke_AFS[it,2,1] = entot(vx_A, vy_A, vz_A)
        me_ke_AFS[it,2,2] = entot(vx_F, vy_F, vz_F)
        me_ke_AFS[it,2,3] = entot(vx_S, vy_S, vz_S)

    end

    return me_ke_AFS
end


function sf2d(bx::Cube, by::Cube, bz::Cube, nres::Integer, nrandpts::Integer, n_avg_bfield_pts::Integer)
    
    nresby4 = div(nres, 4)
    sf_2d = zeros(Float64, nresby4, nresby4)
    
    for (ipar, lpar) in enumerate(0:nresby4-1), (iperp,lperp) in enumerate(0:nresby4-1)
        ssff = 0.0
        numpt = 0

        for kup in 1:nrandpts

            ri = rand(1:nres,3)

            sp_rad = sqrt(lpar.^2 .+ lperp^2) / 2.0
            # Find all points inside a sphere of radius sp_rad centered at ri
            inside_sphere_indices = []
            for ix in max(1, ri[1]-ceil(Int, sp_rad)):min(nres, ri[1]+ceil(Int, sp_rad))
                for iy in max(1, ri[2]-ceil(Int, sp_rad)):min(nres, ri[2]+ceil(Int, sp_rad))
                    for iz in max(1, ri[3]-ceil(Int, sp_rad)):min(nres, ri[3]+ceil(Int, sp_rad))
                        # Calculate distance from ri
                        dist = sqrt((ix-ri[1])^2 + (iy-ri[2])^2 + (iz-ri[3])^2)
                        if dist <= sp_rad
                            push!(inside_sphere_indices, (ix, iy, iz))
                        end
                    end
                end
            end

            # If no points found (unlikely but possible), use a single point approach
            if isempty(inside_sphere_indices)
                bhat = [bx[ri[1], ri[2], ri[3]], by[ri[1], ri[2], ri[3]], bz[ri[1], ri[2], ri[3]]]
            else
                # Calculate mean magnetic field inside the sphere
                bx_sum, by_sum, bz_sum = 0.0, 0.0, 0.0
                for (ix, iy, iz) in inside_sphere_indices
                    bx_sum += bx[ix, iy, iz]
                    by_sum += by[ix, iy, iz]
                    bz_sum += bz[ix, iy, iz]
                end
                
                n_points = length(inside_sphere_indices)
                bhat = [bx_sum/n_points, by_sum/n_points, bz_sum/n_points]
            end

            # Normalize to get unit vector
            bhat = bhat / (sqrt(sum(bhat.^2)))

            vrand = rand(3)

            perpb_x = bhat[2] * vrand[3] - bhat[3] * vrand[2]
            perpb_y = bhat[3] * vrand[1] - bhat[1] * vrand[3]
            perpb_z = bhat[1] * vrand[2] - bhat[2] * vrand[1]

            perpb = [ perpb_x, perpb_y, perpb_z ]
            perpb = perpb / (sqrt(sum(perpb.^2)))

            r1 = Int.(floor.(ri + (lperp / 2.0) .* perpb + (lpar / 2.0) .* bhat))
            r2 = Int.(floor.(ri - (lperp / 2.0) .* perpb - (lpar / 2.0) .* bhat))

            r1 = 1 .+ mod.(r1, nres)
            r2 = 1 .+ mod.(r2, nres)

            b1 = [ bx[r1[1],r1[2],r1[3]], by[r1[1],r1[2],r1[3]], bz[r1[1],r1[2],r1[3]] ]
            b2 = [ bx[r2[1],r2[2],r2[3]], by[r2[1],r2[2],r2[3]], bz[r2[1],r2[2],r2[3]] ]

            diff = b1 - b2
            summb = sum(diff.^2)
            if isinf(summb) || isnan(summb)
                print("isinf isnan")
                continue
            end

            ssff += summb
            numpt += 1

        end
        sf_2d[ipar, iperp] = ssff / numpt
    end
    return sf_2d
end


function structure_function_2d(vx::Cube, vy::Cube, vz::Cube, max_separation::Int, num_samples::Int)
    nx, ny, nz = size(vx)
    sf = zeros(Float64, max_separation, max_separation)
    counts = zeros(Int, max_separation, max_separation)
    
    # Compute mean magnetic field direction
    bx_mean, by_mean, bz_mean = mean(vx), mean(vy), mean(vz)
    b_direction = [bx_mean, by_mean, bz_mean]
    b_mag = norm(b_direction)
    
    # Normalize to get unit vector (if non-zero)
    if b_mag > 1e-10
        b_direction = b_direction / b_mag
    else
        # Default direction if mean field is near zero
        b_direction = [0.0, 0.0, 1.0]
    end
    
    # Compute perpendicular direction
    perp_direction = [0.0, 0.0, 0.0]
    if abs(b_direction[3]) < 0.9
        perp_direction = normalize([b_direction[2], -b_direction[1], 0.0])
    else
        perp_direction = normalize([1.0, 0.0, 0.0])
    end
    
    for _ in 1:num_samples
        # Random starting point
        x, y, z = rand(1:nx), rand(1:ny), rand(1:nz)
        
        for l_par in 1:max_separation, l_perp in 1:max_separation
            # Skip the zero displacement case
            (l_par == 1 && l_perp == 1) && continue
            
            # Calculate displacement vectors
            δpar = (l_par - 1) * b_direction
            δperp = (l_perp - 1) * perp_direction
            
            # Calculate endpoint coordinates with periodic boundary conditions
            x2 = mod1(round(Int, x + δpar[1] + δperp[1]), nx)
            y2 = mod1(round(Int, y + δpar[2] + δperp[2]), ny)
            z2 = mod1(round(Int, z + δpar[3] + δperp[3]), nz)
            
            # Calculate squared difference for vector components
            diff_squared = (vx[x,y,z] - vx[x2,y2,z2])^2 + 
                          (vy[x,y,z] - vy[x2,y2,z2])^2 + 
                          (vz[x,y,z] - vz[x2,y2,z2])^2
            
            # Accumulate result
            sf[l_par, l_perp] += diff_squared
            counts[l_par, l_perp] += 1
        end
    end
    
    # Normalize by counts
    for i in eachindex(sf)
        if counts[i] > 0
            sf[i] /= counts[i]
        end
    end
    
    return sf
end

function pairwise_annular_average(map::Matx, inner_radius::Int, outer_radius::Int)::Matx
    result = similar(map)
    
    for i in 1:size(map, 1)
        for j in 1:size(map, 2)
            # Region of interest (ROI) based on inner and outer limit
            inner_square = max(1, i - inner_radius)
            outer_square = min(size(map, 1), i + outer_radius)
            
            inner_radius_squared = inner_radius^2
            outer_radius_squared = outer_radius^2
            
            # Calculate the pairwise annular average for the current pixel
            total_sum = zero(typeof(map[1,1]))
            count = 0
            
            for x in inner_square:outer_square
                for y in max(1, j - outer_radius):min(size(map, 2), j + outer_radius)
                    # Check if the indices are within the valid range
                    if x >= 1 && x <= size(map, 1) && y >= 1 && y <= size(map, 2)
                        distance_squared = (i - x)^2 + (j - y)^2
                        #print(distance_squared)
                        
                        if distance_squared >= inner_radius_squared && distance_squared <= outer_radius_squared
                            total_sum += map[x, y]
                            count += 1
                        end
                    end
                end
            end
            
            # Calculate the average and store it in the result matrix
            result[i, j] = count == 0 ? zero(typeof(map[1,1])) : total_sum / count
        end
    end
    
    return result
end

function kurtosis_3d(iv, jv, kv)
    nx, ny, nz = size(iv)
    max_sep = div(nx, 2)
    
    # Initialize structure function arrays for each velocity component
    kurt_vol = zeros(Float64, div(max_sep, 2))
    
    for (i, dr) in enumerate(1:2:max_sep)

            #du_x = circshift(iv, (-dr, 0, 0)) .- iv
            #dv_x = circshift(jv, (-dr, 0, 0)) .- jv
            #dw_x = circshift(kv, (-dr, 0, 0)) .- kv

            du_y = circshift(iv, (0, -dr, 0)) .- iv
            dv_y = circshift(jv, (0, -dr, 0)) .- jv
            dw_y = circshift(kv, (0, -dr, 0)) .- kv

     	    ds2y = du_y.^2 .+ dv_y.^2 .+ dw_y.^2
            ds4y = du_y.^4 .+ dv_y.^4 .+ dw_y.^4

	    du_y = dv_y = dw_y = nothing;
	    GC.gc()

	    kurt_y = mean(ds4y) ./ mean(ds2y).^2

            du_z = circshift(iv, (0, 0, -dr)) .- iv
            dv_z = circshift(jv, (0, 0, -dr)) .- jv
            dw_z = circshift(kv, (0, 0, -dr)) .- kv

	    ds2z = du_z.^2 .+ dv_z.^2 .+ dw_z.^2
	    ds4z = du_z.^4 .+ dv_z.^4 .+ dw_z.^4

	    dy_z = dv_z = dw_z = nothing;
	    GC.gc()
	    
	    kurt_z = mean(ds4z) ./ mean(ds2z).^2

	    kurt_t = (kurt_y + kurt_z) / 2
	    #kurt_vol[i] = sum(kurt_t .> 3.0)
	    kurt_vol[i] = kurt_t

    end
    
    return kurt_vol
end


end #top
