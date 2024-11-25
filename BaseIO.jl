#***********************************************************************
module BaseIO
#***********************************************************************

# MATH PACKAGES
using Statistics
using FFTW, Random
using LinearAlgebra

# DATA PACKAGES
using HDF5

# CUSTOM PACKAGES
using .DataStruct

#***********************************************************************

export hread, read_data_unified, read_pencil_hdf
export rs_kij, read_gsa, convert_athdf, read_data_athdf
export read_data_athdf_bonly, read_data_athdf_dvonly

#***********************************************************************


function hread(f, db)
    return convert(Array{Float64,3}, read(f,db))
end

function convert_athdf(db,Nx::Int,Ny::Int,Nz::Int,mode)
	f = h5open(db,"r")
	prim = read(f,"prim")
 	LL   = read(f,"LogicalLocations")
 	Cubes = Dict("d"=>init_array(Float64, Nx,Ny,Nz) ,"p"=>init_array(Float64, Nx,Ny,Nz) ,"G"=>init_array(Float64, Nx,Ny,Nz),
 		         "ib"=>init_array(Float64, Nx,Ny,Nz),"jb"=>init_array(Float64, Nx,Ny,Nz),"kb"=>init_array(Float64, Nx,Ny,Nz),
 		         "iv"=>init_array(Float64, Nx,Ny,Nz),"jv"=>init_array(Float64, Nx,Ny,Nz),"kv"=>init_array(Float64, Nx,Ny,Nz))
 	Cubes["d"] = puzzleX(prim,LL,1,Nx,Ny,Nz)
 	if (mode.EOS == "Isothermal" )
		Cubes["iv"] = puzzleX(prim,LL,2,Nx,Ny,Nz)
		Cubes["jv"] = puzzleX(prim,LL,3,Nx,Ny,Nz)
		Cubes["kv"] = puzzleX(prim,LL,4,Nx,Ny,Nz)
 	elseif (mode.EOS == "Adiabatic")
		Cubes["p"]  = puzzleX(prim,LL,2,Nx,Ny,Nz)
		Cubes["iv"] = puzzleX(prim,LL,3,Nx,Ny,Nz)
		Cubes["jv"] = puzzleX(prim,LL,4,Nx,Ny,Nz)
		Cubes["kv"] = puzzleX(prim,LL,5,Nx,Ny,Nz)
	end
	if ( mode.B == true )
 		B = read(f,"B")
		Cubes["ib"] = puzzleX(B,LL,1,Nx,Ny,Nz)
		Cubes["jb"] = puzzleX(B,LL,2,Nx,Ny,Nz)
		Cubes["kb"] = puzzleX(B,LL,3,Nx,Ny,Nz)
	end
	if (mode.G == true)
		Cubes["G"] = puzzleX(prim,LL,size(prim)[end],Nx,Ny,Nz)
	end
	close(f)
	return Cubes
end



function read_data_athdf(db, nsize)
    f = h5open(db,"r")
    prim = read(f,"prim")
    LL   = read(f,"LogicalLocations")
    
    d =Float64.(puzzleX(prim,LL,1,nsize,nsize,nsize))
    iv=Float64.(puzzleX(prim,LL,2,nsize,nsize,nsize))
    jv=Float64.(puzzleX(prim,LL,3,nsize,nsize,nsize))
    kv=Float64.(puzzleX(prim,LL,4,nsize,nsize,nsize))
    
    B = read(f,"B")
    
    ib=Float64.(puzzleX(B,LL,1,nsize,nsize,nsize))
    jb=Float64.(puzzleX(B,LL,2,nsize,nsize,nsize))
    kb=Float64.(puzzleX(B,LL,3,nsize,nsize,nsize))

    close(f)
    
    return d,iv,jv,kv,ib,jb,kb
end


function read_data_athdf_dvonly(db, nsize)
    
    f = h5open(db,"r")
    prim = read(f,"prim")
    LL   = read(f,"LogicalLocations")
    
    d =puzzleX(prim,LL,1,nsize,nsize,nsize)
    iv=puzzleX(prim,LL,2,nsize,nsize,nsize)
    jv=puzzleX(prim,LL,3,nsize,nsize,nsize)
    kv=puzzleX(prim,LL,4,nsize,nsize,nsize)
    
    close(f)
    
    return d,iv,jv,kv
end


function read_data_athdf_bonly(db, nsize)
    
    f = h5open(db,"r")
    LL = read(f,"LogicalLocations")
   
    B = read(f,"B")
    
    ib=puzzleX(B,LL,1,nsize,nsize,nsize)
    jb=puzzleX(B,LL,2,nsize,nsize,nsize)
    kb=puzzleX(B,LL,3,nsize,nsize,nsize)

    close(f)
    
    return ib,jb,kb
end


function read_data_athdf_bonly_gc(db, nsize)
    
    f = h5open(db,"r");GC.gc(true)
    LL   = read(f,"LogicalLocations");GC.gc(true)
   
    B = read(f,"B")
    
    ib=puzzleX(B,LL,1,nsize,nsize,nsize)
    jb=puzzleX(B,LL,2,nsize,nsize,nsize)
    kb=puzzleX(B,LL,3,nsize,nsize,nsize)

    close(f)
    
    return ib,jb,kb
end

function read_data_unified(db;ftype="Z",nsize=512)
    if (ftype=="Z")
        f=h5open(db)
        d=hread(f,"gas_density")
        ib=hread(f,"i_mag_field")
        jb=hread(f,"j_mag_field")
        kb=hread(f,"k_mag_field")
        iv=hread(f,"i_velocity")
        jv=hread(f,"j_velocity")
        kv=hread(f,"k_velocity")
    elseif (ftype=="A")
        athsize=nsize
        modess = decompression_mode("Isothermal",true,true)
        x=convert_athdf(db,athsize,athsize,athsize,modess)
        d=x["d"]
        iv=x["iv"]
        jv=x["jv"]
        kv=x["kv"]
        ib=x["ib"]
        jb=x["jb"]
        kb=x["kb"]
    end
    return d,iv,jv,kv,ib,jb,kb
end


function read_pencil_hdf(fname::String; reorder_kij::Bool=false)
    
    f=h5open(fname)
    dn = hread(f,"dn")
    bi = hread(f,"bx")
    bj = hread(f,"by")
    bk = hread(f,"bz")
    vi = hread(f,"vx")
    vj = hread(f,"vy")
    vk = hread(f,"vz")
    
    if reorder_kij
        return rs_kij(dn), rs_kij(bi), rs_kij(bj), rs_kij(bk), rs_kij(vi), rs_kij(vj), rs_kij(vk) 
    else
        return dn, bi, bj, bk, vi, vj, vk 
    end
end


function rs_kij(bb::Cube)
    return permutedims(bb, (3,1,2))
end


function read_gsa(fname::String)
    
    f=h5open(fname)
    gsa_array=hread(f,"c2")
        
    return gsa_array
end

function puzzleX(prim,LL,k,Nx,Ny,Nz)
    c=map(Float64, init_array(Float64, Nx,Ny,Nz))
    dN = [size(prim)[1];size(prim)[2];size(prim)[3]]
    for i = 1:size(LL)[2]
        x1s, x2s, x3s = (LL[:, i] .* dN) .+ 1
        x1f, x2f, x3f = (LL[:, i] .+ 1) .* dN
        c[x1s:x1f, x2s:x2f, x3s:x3f] = prim[:, :, :, i, k]
    end
    return c
end


end #top
