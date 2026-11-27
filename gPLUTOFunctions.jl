#***********************************************************************
module gPLUTOFunctions
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
using DecompMHD

# PLOTTING PACKAGES
using PythonPlot, LaTeXStrings, PythonCall
const plt = PythonPlot
const mpl = pyimport("matplotlib")

#***********************************************************************

export gpluto_get_nvar
export gpluto_get_pfields_dbl, gpluto_get_pfields_flt
export gpluto_read_fluid_bin, gpluto_read_fluid_bin_nvar
export gpluto_read_particles_header, gpluto_read_particles_bin
export gpluto_vis_gif_imshow2D
export gpluto_get_kinetic_energy_density
export gpluto_get_magnetic_energy_density
export gpluto_get_thermal_energy_density
export gpluto_particles_integrate_spectra

#***********************************************************************

"""
    Module gPLUTOFunctions

Functions for gPLUTO testing
"""

#***********************************************************************

function gpluto_get_nvar()
    rho = 1
    vx1 = 2
    vx2 = 3
    vx3 = 4
    bx1 = 5
    bx2 = 6
    bx3 = 7
    prs = 8
    glm = 9
    return (; rho, vx1, vx2, vx3, bx1, bx2, bx3, prs, glm)
end


function gpluto_get_pfields_dbl()
    pid = 1:2
    ppos = 3:5
    ptinj = 6:6
    pcolor = 7:12
    pchi = 13:140
    peng = 141:269
    pnmicro = 270:270
    pdensity = 271:271
    ppressure = 272:272
    plorg = 273:273
    pcr = 274:274
    pflag = 275:275
    pflagprv = 276:276
    pshkgrad = 277:277
    pvu = 278:286
    pvd = 287:295
    return (; pid, ppos, ptinj, pcolor, pchi, peng, pnmicro, pdensity, ppressure, plorg, pcr, pflag, pflagprv, pshkgrad, pvu, pvd)
end

function gpluto_get_pfields_flt()
    pid = 1:2
    ppos = 3:5
    ptinj = 6:6
    pcolor = 7:12
    pchi = 13:140
    peng = 141:269
    pnmicro = 270:270
    pdensity = 271:271
    ppressure = 272:272
    plorg = 273:273
    pshkgrad = 274:274
    return (; pid, ppos, ptinj, pcolor, pchi, peng, pnmicro, pdensity, ppressure, plorg, pshkgrad)
end



function gpluto_read_fluid_bin(file_dir::String, timestep::Int, nx::Int, ny::Int, nz::Int; div_clean::Bool=false)
    filename = file_dir * "/data." * lpad(timestep, 4, "0") * ".dbl"

    if div_clean
        data_read = Array{Float64}(undef, nx, ny, nz, 9)
    else
        data_read = Array{Float64}(undef, nx, ny, nz, 8)
    end

    open(filename, "r") do io
        read!(io, data_read)
    end
    return data_read
    GC.gc()
end

function gpluto_read_fluid_bin(file_dir::String, timestep::Int, nx::Int; div_clean::Bool=false)
    gpluto_read_fluid_bin(file_dir, timestep, nx, nx, nx, div_clean=div_clean)
end

function gpluto_read_fluid_bin_nvar(file_dir::String, timestep::Int, nx::Int, ny::Int, nz::Int, nvar::String)
    filename = file_dir * "/data." * lpad(timestep, 4, "0") * ".dbl"
    
    nvar_list = gpluto_get_nvar()
    nvar_index = getproperty(nvar_list, Symbol(nvar))

    data_read = Array{Float64}(undef, nx, ny, nz)

    open(filename, "r") do io
        for nvar_loop in 1:nvar_index
            read!(io, data_read)
        end
    end
    return data_read
    GC.gc()
end

function gpluto_read_fluid_bin_nvar(file_dir::String, timestep::Int, nx::Int, nvar::String)
    gpluto_read_fluid_bin_nvar(file_dir, timestep, nx, nx, nx, nvar)
end


function gpluto_read_particles_header(file_dir::String, timestep::Int, precision::String)
    
    if precision == "double"
        file_ext = ".dbl" 
    elseif precision == "single"
        file_ext = ".flt"
    end
    
    filename = file_dir * "/particles.0" * lpad(timestep, 3, "0") * "_ch00" * file_ext


    header_fields = ["info", "dimensions", 
                     "nflux", "invDt_particles", 
                     "endianity", "Totparticles", 
                     "nparticles (per chunk)", "nprocs", 
                     "npart_rank", "decomposition", 
                     "nchunks", "idCounter", 
                     "lastInj", "particletype", 
                     "precision", "time", 
                     "stepNumber", "nfields", 
                     "field_names", "field_dim"]
                 
    string_fields = [5, 10, 15, 19, 20]
    nheaderfields = length(header_fields)

    header = Array{Any}(undef, nheaderfields, 2)
    header[:, 1] = header_fields
    # header[10, 2] = Vector{String}(undef, 3)
    
    dims = Int[]
    nfield_dims = 0

    open(filename, "r") do io
        for i in 1:nheaderfields
            if i == 1
                header[i, 2] = strip(readline(io)[3:end])
            elseif i in string_fields
                header[i, 2] = strip(readline(io))[21:end]
            # elseif i == 10
            #     tmp_str = strip(readline(io))[21:33]
            #     header[i, 2][:] .= tmp_str[1:3], tmp_str[6:8], tmp_str[11:13]
            else 
                header[i, 2] = parse(Float64, strip(readline(io))[21:end])
            end
        end
        dims = parse.(Int, split(strip(header[20, 2])))
        nfield_dims = sum(dims)
    end
    return header, nfield_dims, nheaderfields, file_ext
end



function gpluto_read_particles_bin(file_dir::String, timestep::Int; get_header::Bool=false, precision::String="double")
    
    header, nfield_dims, nheaderfields, file_ext = gpluto_read_particles_header(file_dir, timestep, precision)

    # precision = header[15, 2]
    dtype = precision == "single" ? Float32 : Float64

    Totparticles = Int(header[6, 2])
    nchunks = Int(header[11, 2])
    nparticles = zeros(Int, nchunks)

    data_merged = Array{dtype}(undef, nfield_dims, Totparticles)

    for chunk in 0:nchunks-1
        chunk_filename = file_dir * "/particles.0" * lpad(timestep, 3, "0") * "_ch" * lpad(chunk, 2, "0") * file_ext

        open(chunk_filename, "r") do io

            for i in 1:6
                _ = readline(io)
            end
            nparticles_chunk = parse(Int, strip(readline(io))[21:end])
            for i in 1:nheaderfields-7
                _ = readline(io)
            end
            
            data_chunk = Array{dtype}(undef, nfield_dims, nparticles_chunk)
            read!(io, data_chunk)

            if chunk == 0
                data_merged[:, 1:nparticles_chunk] = data_chunk
            else
                data_merged[:, sum(nparticles[1:chunk]) + 1:sum(nparticles[1:chunk]) + nparticles_chunk] = data_chunk
            end
            nparticles[chunk + 1] = nparticles_chunk
        end
    end
    header[7, 2] = nparticles

    machine_endian = ENDIAN_BOM == 0x04030201 ? "little" : "big"
    file_endian = header[5, 2]
    if file_endian != machine_endian
        bswap!(data_merged)
    end

    if get_header
        return header, Float64.(data_merged')
    else
        return Float64.(data_merged')
    end
    GC.gc()
end

function gpluto_vis_gif_imshow2D(file_dir::String, primv::String, nx::Int, frames::UnitRange{Int}, 
    zslice::UnitRange{Int}, vmin::Float64, vmax::Float64; fps::Int=4, cmap::String="magma_r", 
    out_dir::String="./", logscale::Bool=false, tnorm::Float64=1.0, tunit::String="", 
    titstring::AbstractString="")

    fig, ax = plt.subplots()
    rho0 = gpluto_read_fluid_bin_nvar(file_dir, 0, nx, primv)
    
    frame0 = logscale ? log10.(mean(rho0[:, :, zslice], dims=3)) : (mean(rho0[:, :, zslice], dims=3))

    im = ax.imshow(frame0, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax)

    function update(tstep)
        rho_in = gpluto_read_fluid_bin_nvar(file_dir, tstep, nx, primv)

        frame = logscale ? log10.(mean(rho_in[:, :, zslice], dims=3)) : (mean(rho_in[:, :, zslice], dims=3))
        im.set_data(frame)
        # im.set_clim(minimum(rho_in), maximum(rho_in))
        ax.set_title(titstring * "t = " * rpad(round(tstep * tnorm, digits=2), 4, "0") * " " * tunit)
        return im
    end

    anim = pyimport("matplotlib.animation").FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000/fps,
        blit=false
    )

    anim.save(out_dir * "anim_" * primv * ".gif", writer="pillow", fps=fps);
    frame = frame0 = nothing
    GC.gc()
end

function gpluto_get_kinetic_energy_density(glob_data, tstep, nx, ny, nz)

    vx1 = gpluto_read_fluid_bin_nvar(glob_data, tstep, nx, ny, nz, "vx1");
    vx2 = gpluto_read_fluid_bin_nvar(glob_data, tstep, nx, ny, nz, "vx2");
    vx3 = gpluto_read_fluid_bin_nvar(glob_data, tstep, nx, ny, nz, "vx3");

    rho = gpluto_read_fluid_bin_nvar(glob_data, tstep, nx, ny, nz, "rho");

    kin_energy_density = 0.5 * mean(rho .* (vx1.^2 .+ vx2.^2 .+ vx3.^2));

    return kin_energy_density
end

function gpluto_get_kinetic_energy_density(glob_data, tstep, nx)
    return gpluto_get_kinetic_energy_density(glob_data, tstep, nx, nx, nx)
end

function gpluto_get_magnetic_energy_density(glob_data, tstep, nx, ny, nz)

    bx1 = gpluto_read_fluid_bin_nvar(glob_data, tstep, nx, ny, nz, "bx1");
    bx2 = gpluto_read_fluid_bin_nvar(glob_data, tstep, nx, ny, nz, "bx2");
    bx3 = gpluto_read_fluid_bin_nvar(glob_data, tstep, nx, ny, nz, "bx3");

    mag_energy_density = mean((bx1.^2 .+ bx2.^2 .+ bx3.^2) ./ (8.0 * π));

    return mag_energy_density
end

function gpluto_get_magnetic_energy_density(glob_data, tstep, nx)
    return gpluto_get_magnetic_energy_density(glob_data, tstep, nx, nx, nx)
end

function gpluto_get_thermal_energy_density(glob_data, tstep, nx, ny, nz)

    prs = gpluto_read_fluid_bin_nvar(glob_data, tstep, nx, ny, nz, "prs");

    thermal_energy_density = mean((prs) ./ (2.0 / 3.0));

    return thermal_energy_density
end

function gpluto_get_thermal_energy_density(glob_data, tstep, nx)
    return gpluto_get_thermal_energy_density(glob_data, tstep, nx, nx, nx)
end

function gpluto_particles_integrate_spectra(chi_np::Vect, eng_np::Vect, density_np::Float64)::Float64
    Neng_sum = 0.0
    
    # chi_np = chi(pc, np)
    # eng_np = eng(pc, np)
    
    for e in 1:128
        scrh = chi_np[e] * (eng_np[e + 1] - eng_np[e])
        Emid = 0.5 * (eng_np[e] + eng_np[e + 1])
        Neng_sum += scrh * Emid
    end
    
    return Neng_sum * density_np
end



end # module gPLUTOFunctions
