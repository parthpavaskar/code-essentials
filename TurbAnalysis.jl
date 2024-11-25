#***********************************************************************
module TurbAnalysis
#***********************************************************************

# MATH PACKAGES
using Statistics
using FFTW, Random
using LinearAlgebra
using Images

# CUSTOM PACKAGES
using .DataStruct
using .BaseIO
using .ArrayFunctions
using .DecompMHD

#***********************************************************************

export oned_energy_spectrum, oned_energy_spectrum_para, oned_energy_spectrum_perp
export PCASF_decomposition
export decomp_simple_A, decomp_simple_C, decomp_simple_F, decomp_simple_S
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
    kx = [i-(nx/2) for i in 0:nx-1]
    ky = [j-(ny/2) for j in 0:ny-1]
    kz = [k-(nz/2) for k in 0:nz-1]

    ksq = [kx[i]^2 + ky[j]^2 + kz[k]^2 for i in 1:nx, j in 1:ny, k in 1:nz]
    kmax = maximum(ksq)
    kbins = round.(Int, sqrt.(ksq))

    maxbin = maximum(kbins)
    Pk = zeros(maxbin + 1)

    for k in 1:nz, j in 1:ny, i in 1:nx
        bin = kbins[i, j, k]
        Pk[bin + 1] += power_spectrum[i, j, k]
    end
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
    kx = [i-(nx/2) for i in 0:nx-1]
    ky = [j-(ny/2) for j in 0:ny-1]
    kz = [k-(nz/2) for k in 0:nz-1]

    ksq = [kx[i]^2 + ky[j]^2 + kz[k]^2 for i in 1:nx, j in 1:ny, k in 1:nz]
    kmax = maximum(ksq)
    kbins = round.(Int, sqrt.(ksq))

    maxbin = maximum(kbins)
    Pk = zeros(maxbin + 1)

    for k in 1:nz, j in 1:ny, i in 1:nx
        bin = kbins[i, j, k]
        Pk[bin + 1] += power_spectrum[i, j, k]
    end

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
    kx = [i-(nx/2) for i in 0:nx-1]
    ky = [j-(ny/2) for j in 0:ny-1]
    kz = [k-(nz/2) for k in 0:nz-1]

    ksq = [kx[i]^2 + ky[j]^2 + kz[k]^2 for i in 1:nx, j in 1:ny, k in 1:nz]
    kmax = maximum(ksq)
    kbins = round.(Int, sqrt.(ksq))

    maxbin = maximum(kbins)
    Pk = zeros(maxbin + 1)

    for k in 1:nz, j in 1:ny, i in 1:nx
        bin = kbins[i, j, k]
        Pk[bin + 1] += power_spectrum[i, j, k]
    end
    return Pk
end


function oned_energy_spectrum_para(u::Cube, v::Cube, w::Cube)::Vect
    
    # Perform FFT on each component of the field
    u_fft = fft(u)
    v_fft = fft(v)
    w_fft = fft(w)

    # Compute the power spectrum
    power_spectrum = fftshift(abs2.(u_fft) + abs2.(v_fft) + abs2.(w_fft))
    
    # Average over spherical shells
    nx, ny, nz = size(u)
    kz = [k-(nz/2) for k in 0:nz-1]

    ksq = [kz[k]^2 for k in 1:nz]
    kmax = maximum(ksq)
    kbins = round.(Int, sqrt.(ksq))

    maxbin = maximum(kbins)
    Pk = zeros(maxbin + 1)

    power_spectrum_sum = sum(power_spectrum, dims=(1, 2))
    for k in 1:nz
        bin = kbins[k]
        Pk[bin + 1] += power_spectrum_sum[k]
    end
    return Pk
end



function oned_energy_spectrum_perp(u::Cube, v::Cube, w::Cube)::Vect
    
    # Perform FFT on each component of the field
    u_fft = fft(u)
    v_fft = fft(v)
    w_fft = fft(w)

    # Compute the power spectrum
    power_spectrum = fftshift(abs2.(u_fft) + abs2.(v_fft) + abs2.(w_fft))
    
    # Average over spherical shells
    nx, ny, nz = size(u)
    kx = [i-(nx/2) for i in 0:nx-1]
    ky = [j-(ny/2) for j in 0:ny-1]

    ksq = [kx[i]^2 + ky[j]^2 for i in 1:nx, j in 1:ny]
    kmax = maximum(ksq)
    kbins = round.(Int, sqrt.(ksq))

    maxbin = maximum(kbins)
    Pk = zeros(maxbin + 1)

    power_spectrum_sum = sum(power_spectrum, dims=3)
    for j in 1:ny, i in 1:nx
        bin = kbins[i, j]
        Pk[bin + 1] += power_spectrum_sum[i, j]
    end
    return Pk
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

function enfrac_vs_time(data_vec::Vector{String}, nres::Integer, v_A=1.0::Float64, c_s=1.0::Float64; resize_factor=4::Integer)::Array{Float64,3}
    
    n_resize = div(nres, resize_factor)
    me_ke_AFS = init_array(Float64, length(data_vec), 2, 3)

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

        vx_A, vy_A, vz_A, bx_A, by_A, bz_A = decomposition_A(vx, vy, vz, bx, by, bz, v_A, c_s, ρ_0=rho_0)
        vx_F, vy_F, vz_F, bx_F, by_F, bz_F = decomposition_F(vx, vy, vz, bx, by, bz, v_A, c_s, ρ_0=rho_0)
        vx_S, vy_S, vz_S, bx_S, by_S, bz_S = decomposition_S(vx, vy, vz, bx, by, bz, v_A, c_s, ρ_0=rho_0)

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
