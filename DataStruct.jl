#***********************************************************************
module DataStruct
#***********************************************************************

export Vect, Matx, Cube
export init_array, decompression_mode

#***********************************************************************

const Vect{T}  = Array{T,1}
const Matx{T}  = Array{T,2}
const Cube{T}  = Array{T,3}

function init_array(::Type{T}, dims::Integer...) where T
    return Array{T}(undef, dims)
end

struct decompression_mode
    EOS
    B
    G
end

end #top
