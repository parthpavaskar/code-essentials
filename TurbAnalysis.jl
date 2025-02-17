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

export oned_energy_spectrum, oned_energy_spectrum_para, oned_energy_spectrum_perp
export twod_energy_spectrum, fourier_filter_2D, fourier_filter_3D
export get_ma_dv, get_ma_db, get_ma2d_dv, get_ma2d_db, get_ms, entot, entot_weighted
export get_enfrac_ac, get_enfrac_asf, get_enfrac_asf_total, enfrac_vs_time
export sf2d

#***********************************************************************

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
    power_spectrum_sum = sum(power_spectrum, dims=3)
    
    # Average over spherical shells
    nx, ny, nz = size(u)
    kx = collect(0:nx-1) .- (nx / 2)
    ky = collect(0:ny-1) .- (ny / 2)
    kx_g, ky_g = meshgrid_2D(kx, ky)
    
    ksq = kx_g.^2 .+ ky_g.^2
    kbins = round.(Int, sqrt.(ksq))

    Pk = zeros(nz, maximum(kbins) + 1)
    [Pk[k, kbins[i,j] + 1] += power_spectrum[i,j,k] for k in 1:nz, j in 1:ny, i in 1:nx]

    return log.(Pk[div(nz,2)+2:end, 2:div(nx,2)])
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
    sf_2d = init_array(Float64, nresby4, nresby4)
    
    for ipar in 1:nresby4
    
        lpar = ipar .* 1.0 - 1

        for iperp in 1:nresby4

            lperp = iperp * 1.0 - 1 
            
            ssff = 0.0
            numpt = 0

            for kup in 1:nrandpts

                ri = rand(1:nres,3)

                sp_rad = sqrt(lpar.^2 .+ lperp^2) / 2.0

                lr = rand(n_avg_bfield_pts) .* sp_rad
                theta = rand(n_avg_bfield_pts) .* π
                phi   = rand(n_avg_bfield_pts) .* 2.0 .* π

                lx = lr .* sin.(theta) .* cos.(phi)
                ly = lr .* sin.(theta) .* sin.(phi)
                lz = lr .* cos.(theta)

                xis = Int.(floor.(ri[1] .+ lx))
                yis = Int.(floor.(ri[2] .+ ly))
                zis = Int.(floor.(ri[3] .+ lz))
                
                xis = 1 .+ mod.(xis, nres)
                yis = 1 .+ mod.(yis, nres)
                zis = 1 .+ mod.(zis, nres)
                
                bxs = [bx[xis[ii],yis[ii],zis[ii]] for ii in 1:n_avg_bfield_pts]
                bys = [by[xis[ii],yis[ii],zis[ii]] for ii in 1:n_avg_bfield_pts]
                bzs = [bz[xis[ii],yis[ii],zis[ii]] for ii in 1:n_avg_bfield_pts]
                
                bhat = [ mean(bxs), mean(bys), mean(bzs) ]
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
    end
    return sf_2d
end



end #top
