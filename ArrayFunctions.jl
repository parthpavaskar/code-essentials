#***********************************************************************
module ArrayFunctions
#***********************************************************************

# MATH PACKAGES
using Statistics
using FFTW, Random
using LinearAlgebra

# CUSTOM PACKAGES
using DataStruct
using BaseIO

#***********************************************************************

export meshgrid_2D, meshgrid_3D
export sub_grid, sub_blocks, select_grid, cstd
export outer_product, identity_matrix, vector_triple_product
export vector_rotation, scalar_rotation
export add_periodic_ghost_cells

#***********************************************************************


function meshgrid_2D(v1::Vect, v2::Vect)::Tuple{Matx, Matx}
    x, y = length(v1), length(v2)
    vg1 = reshape(v1, x, 1) .* ones(x, y)
    vg2 = reshape(v2, 1, y) .* ones(x, y)
    return vg1, vg2
end

function meshgrid_3D(v1::Vect, v2::Vect, v3::Vect)::Tuple{Cube, Cube, Cube}
    x, y, z = length(v1), length(v2), length(v3)
    vg1 = reshape(v1, x, 1, 1) .* ones(x, y, z)
    vg2 = reshape(v2, 1, y, 1) .* ones(x, y, z)
    vg3 = reshape(v3, 1, 1, z) .* ones(x, y, z)
    return vg1, vg2, vg3
end

function sub_grid(arr::Matx, res::Int; center::Bool=true, grid_i::Int=1, grid_j::Int=1)::Matx
    nres = Int(size(arr)[1] / 2)
    if center
        return copy(@view arr[Int(nres+1-(res/2)):Int(nres+(res/2)), Int(nres+1-(res/2)):Int(nres+(res/2))])
    else
        return copy(@view arr[Int((grid_i-1)*res+1):Int(grid_i*res), Int((grid_j-1)*res+1):Int(grid_j*res)])
        return X
    end
end



function sub_blocks(arr::Cube, res::Int; center::Bool=true, grid_i::Int=1, grid_j::Int=1)::Cube
    nres = Int(size(arr)[1] / 2)
    if center
        return copy(@view arr[Int(nres+1-(res/2)):Int(nres+(res/2)), Int(nres+1-(res/2)):Int(nres+(res/2)), :])
    else
        return copy(@view arr[Int((grid_i-1)*res+1):Int(grid_i*res), Int((grid_j-1)*res+1):Int(grid_j*res), :])
    end
end


function select_grid(xx_arr::Matx, spot_rad::Integer, center_i::Int, center_j::Int)::Matx
    return @view xx_arr[center_i-spot_rad+1:center_i+spot_rad, center_j-spot_rad+1:center_j+spot_rad]
end


function cstd(phi::Matx)::Float64
    sph = sin.(phi)
    cph = cos.(phi)
    R = sqrt(mean(sph)^2 + mean(cph)^2)
    return sqrt(-2 * log(R))
end

function outer_product(u::Vect,v::Vect)::Matrix{Float64}
    return u*transpose(v)
end

function identity_matrix(u::Vect)::Matrix{Float64}
    return ones(Float64, length(u), length(u))
end

function vector_triple_product(u::Vect,v::Vect,w::Vect)::Vect
    return cross(u,cross(v,w))
end

function rotation_matrix(θx::Float64, θy::Float64, θz::Float64)::Matrix{Float64}
    Tx = [1 0 0; 0 cos(θx) -sin(θx); 0 sin(θx) cos(θx)]
    Ty = [cos(θy) 0 sin(θy); 0 1 0; -sin(θy) 0 cos(θy)]
    Tz = [cos(θz) -sin(θz) 0; sin(θz) cos(θz) 0; 0 0 1]
    return inv(Tx * Ty * Tz)
end


function volume_scaler_interp(
    x_fl::Int, y_fl::Int, z_fl::Int,
    x_cl::Int, y_cl::Int, z_cl::Int,
    Δx::Float64, Δy::Float64, Δz::Float64, d::Cube)::Float64
    c00 = (1 - Δx) * d[x_fl, y_fl, z_fl] + Δx * d[x_cl, y_fl, z_fl]
    c01 = (1 - Δx) * d[x_fl, y_fl, z_cl] + Δx * d[x_cl, y_fl, z_cl]
    c10 = (1 - Δx) * d[x_fl, y_cl, z_fl] + Δx * d[x_cl, y_cl, z_fl]
    c11 = (1 - Δx) * d[x_cl, y_fl, z_fl] + Δx * d[x_cl, y_cl, z_cl]
    c0  = c00 * (1 - Δy) + c10 * Δy
    c1  = c01 * (1 - Δy) + c11 * Δy
    c   = c0 * (1 - Δz) + c1 * Δz
    return c
end


function check_periodicity!(X::Vect,N::Vect)::Vect
    for k = 1:3
        if X[k] > N[k]
            X[k] -= N[k] - 1
        elseif X[k] < 1
            X[k] += N[k] - 1
        end
    end
    return X
end


function scalar_rotation(d::Cube, θx::Float64, θy::Float64, θz::Float64)::Cube
    nx, ny, nz = size(d)
    rotated_array = init_array(Float64, nx, ny, nz)

    rot_matrix = rotation_matrix(-θx, -θy, -θz)
    coordinates = zeros(3)

    dimensions = [nx, ny, nz]
    NHalf = [nx / 2, ny / 2, nz / 2] .+ 0.5
    
    @inbounds for k = 1:nz
        coordinates[3] = k - NHalf[3]
            for j = 1:ny
                coordinates[2] = j - NHalf[2]
                for i = 1:nx
                    coordinates[1] = i - NHalf[1]
                    X = rot_matrix * coordinates
                    X .+= NHalf
                X = check_periodicity!(X, dimensions)
                x_fl, y_fl, z_fl = Int(floor(X[1])), Int(floor(X[2])), Int(floor(X[3]))
                x_cl, y_cl, z_cl = Int(ceil(X[1])), Int(ceil(X[2])), Int(ceil(X[3]))
                Δx = X[1] - x_fl
                Δy = X[2] - y_fl
                Δz = X[3] - z_fl
                rotated_array[i,j,k] = volume_scaler_interp(x_fl, y_fl, z_fl, x_cl, y_cl, z_cl, Δx, Δy, Δz, d)
            end
        end
    end
    return rotated_array
end



function vector_rotation(iv::Cube, jv::Cube, kv::Cube, θx::Float64, θy::Float64, θz::Float64)::Tuple{Cube, Cube, Cube}
    nx, ny, nz = size(iv)
    rotated_array_i = init_array(Float64, nx, ny, nz)
    rotated_array_j = init_array(Float64, nx, ny, nz)
    rotated_array_k = init_array(Float64, nx, ny, nz)

    rot_matrix = rotation_matrix(-θx, -θy, -θz)
    coordinates = zeros(3)
    
    dimensions = [nx, ny, nz]
    NHalf = [nx / 2, ny / 2, nz / 2] .+ 0.5
    VI = init_array(Float64, 3)
    
    @inbounds for k = 1:nz
    coordinates[3] = k - NHalf[3]
        for j = 1:ny
            coordinates[2] = j - NHalf[2]
            for i = 1:nx
                coordinates[1] = i - NHalf[1]
                X = rot_matrix * coordinates
                X .+= NHalf
                check_periodicity!(X, dimensions)
                x_fl, y_fl, z_fl = Int(floor(X[1])), Int(floor(X[2])), Int(floor(X[3]))
                x_cl, y_cl, z_cl = Int(ceil(X[1])), Int(ceil(X[2])), Int(ceil(X[3]))

                Δx = X[1] - x_fl
                Δy = X[2] - y_fl
                Δz = X[3] - z_fl

                VI[1] = volume_scaler_interp(x_fl, y_fl, z_fl, x_cl, y_cl, z_cl, Δx, Δy, Δz, iv)
                VI[2] = volume_scaler_interp(x_fl, y_fl, z_fl, x_cl, y_cl, z_cl, Δx, Δy, Δz, jv)
                VI[3] = volume_scaler_interp(x_fl, y_fl, z_fl, x_cl, y_cl, z_cl, Δx, Δy, Δz, kv)
                RVI = rot_matrix * VI

                rotated_array_i[i,j,k] = RVI[1]
                rotated_array_j[i,j,k] = RVI[2]
                rotated_array_k[i,j,k] = RVI[3]
            end
        end
    end
    return rotated_array_i, rotated_array_j, rotated_array_k
end


function add_periodic_ghost_cells(array::Cube, num_ghosts::Int)::Cube
    nx, ny, nz = size(array)
    
    ghost_array = init_array(Float64, nx + 2*num_ghosts, ny + 2*num_ghosts, nz + 2*num_ghosts)
    
    @views ghost_array[(num_ghosts+1):(num_ghosts+nx), (num_ghosts+1):(num_ghosts+ny), (num_ghosts+1):(num_ghosts+nz)] .= array

    for g in 1:num_ghosts
        @views ghost_array[num_ghosts+1-g, (num_ghosts+1):(num_ghosts+ny), (num_ghosts+1):(num_ghosts+nz)] .= array[nx-g+1, :, :]
        @views ghost_array[nx+num_ghosts+g, (num_ghosts+1):(num_ghosts+ny), (num_ghosts+1):(num_ghosts+nz)] .= array[g, :, :]

        @views ghost_array[(num_ghosts+1):(num_ghosts+nx), num_ghosts+1-g, (num_ghosts+1):(num_ghosts+nz)] .= ghost_array[(num_ghosts+1):(num_ghosts+nx), ny-g+1, (num_ghosts+1):(num_ghosts+nz)]
        @views ghost_array[(num_ghosts+1):(num_ghosts+nx), ny+num_ghosts+g, (num_ghosts+1):(num_ghosts+nz)] .= ghost_array[(num_ghosts+1):(num_ghosts+nx), g, (num_ghosts+1):(num_ghosts+nz)]

        @views ghost_array[(num_ghosts+1):(num_ghosts+nx), (num_ghosts+1):(num_ghosts+ny), num_ghosts+1-g] .= ghost_array[(num_ghosts+1):(num_ghosts+nx), (num_ghosts+1):(num_ghosts+ny), nz-g+1]
        @views ghost_array[(num_ghosts+1):(num_ghosts+nx), (num_ghosts+1):(num_ghosts+ny), nz+num_ghosts+g] .= ghost_array[(num_ghosts+1):(num_ghosts+nx), (num_ghosts+1):(num_ghosts+ny), g]
    end

    return ghost_array
end


end #top
