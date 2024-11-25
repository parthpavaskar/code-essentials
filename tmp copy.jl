# function for decomposition of MHD modes from MHD 3d simulation data in velocity and B-field space

# LazNewCh5.PCASF_decomposition_v, which is CORRECT for velocity projection
function PCASF_decomposition_b(iv, jv, kv, ib, jb, kb, v_A, c_s; mode="C", epsilon=1/2/size(ib)[1]^2, rho_0=1)
    beta=2*c_s^2/v_A^2

    iv_f=fftshift(fft(iv))
    jv_f=fftshift(fft(jv))
    kv_f=fftshift(fft(kv))

    ibf_o = zeros(typeof(1.0.+im),size(iv_f));
    jbf_o = zeros(typeof(1.0.+im),size(jv_f));                                                                                           
    kbf_o = zeros(typeof(1.0.+im),size(kv_f));
    nx,ny,nz=size(ib);

    mib=mean(ib);
    mjb=mean(jb);                                                                                                                        
    mkb=mean(kb);

    Bhat = [mib,mjb,mkb]

    for i in 1:nx, j in 1:ny, k in 1:nz                                                                                                      
        # get the kk vector
        kx = i - div(nx,2)-1;
        ky = j - div(ny,2)-1;
        kz = k - div(nz,2)-1;
        kk = [kx,ky,kz]
        α=beta/2
        if (sum(kk.^2.0)>0)
            khat = n3(kk) # normalize
            kdotB=sum(kk.*Bhat)./sqrt.(sum(kk.^2.0))./sqrt.(sum(Bhat.^2.0))
            D=(1+α)^2-4*α*kdotB*kdotB
                                                                                                                                                 
            # slow mode
            prefactor1 = 1-sqrt(D)+α;
            prefactor2 = 1+sqrt(D)-α;

            # fast mode                                                                                                                          
            prefactor3 = 1-sqrt(D)-α;
            prefactor4 = 1+sqrt(D)+α;

            if (1.0.-abs.(kdotB) >epsilon)

                if (mode=="A")
                    zeta_o = ucross3(kk,Bhat)
                elseif (mode=="C")
                    zeta_A = ucross3(kk,Bhat)
                    zeta_o = ucross3(kk,zeta_A)
                elseif (mode=="P")
                    zeta_o = n3(kk)

                elseif (mode=="S")                                                                                                                      
                    zeta_ll = n3(Bhat)
                    zeta_pp = n3(kk.-cdot3(kk,zeta_ll).*zeta_ll)
                    zeta_x  = -prefactor2.*zeta_ll.+prefactor1.*zeta_pp
                    if (m3(zeta_x)>epsilon)
                        zeta_o  = n3(zeta_x)                                                                                                             
                    else
                        zeta_o = [0,0,0]
                    end                                                                                                                              
                elseif (mode=="F")
                    zeta_ll = n3(Bhat)
                    zeta_pp = n3(kk.-cdot3(kk,zeta_ll).*zeta_ll)
                    zeta_x  = -prefactor3.*zeta_ll.+prefactor4.*zeta_pp
                    if (m3(zeta_x)>epsilon)
                        zeta_o  = n3(zeta_x)
                    else
                        zeta_o = [0,0,0]
                    end
                end                                                                                                                                  
                # assign a vector at (i,j,k)
                vf = [iv_f[i,j,k],jv_f[i,j,k],kv_f[i,j,k]]
                # do a dot product
                vfdotC = sum(vf.*zeta_o)
                ivf_o= vfdotC.*zeta_o[1]                                                                                                     
                jvf_o= vfdotC.*zeta_o[2]
                kvf_o= vfdotC.*zeta_o[3]
                
                # velocity vector
                vf_o = [ivf_o,jvf_o,kvf_o]
                if (mode == "A" || mode == "S" || mode == "F")
                    c_w=wavespeed(Bhat,khat,v_A,c_s;mode=mode)
                else
                    # temporarily null the two cases
                    c_w = v_A
                end
                # k x (v x B)/c_w = √ρ*v_A/c_w * (khat x (v x Bhat))
                if (c_w>0)
                    ibo,jbo,kbo=double_cross(khat...,vf_o...,Bhat...)/c_w*v_A*sqrt(ρ_0)
                else
                    # mute nonpropagating modes atm.
                    ibo,jbo,kbo=0,0,0;
                end
                ibf_o[i,j,k]=ibo;
                jbf_o[i,j,k]=jbo;
                kbf_o[i,j,k]=kbo;
                # 
            end                                                                                                                              
        end
    end
    ib_o = real(ifft(ifftshift(ibf_o)));
    jb_o = real(ifft(ifftshift(jbf_o)));
    kb_o = real(ifft(ifftshift(kbf_o)));
    return ib_o,jb_o,kb_o
end

function wavespeed(B,k,v_A,c_s;mode="A")
    cosθ=cdot3(n3(B),n3(k))
    if (mode == "A")
        return v_A*cosθ
    else 
        
        Δ2=(v_A^2+c_s^2)^2-4*v_A^2*c_s^2*cosθ^2
        fac=v_A^2+c_s^2
        if (mode == "S")
            return sqrt(0.5*(fac-sqrt(Δ2)))
        elseif (mode == "F")
            return sqrt(0.5*(fac+sqrt(Δ2)))
        end
    end
end



function decomposition(
    iv::Cube, jv::Cube, kv::Cube, 
    ib::Cube, jb::Cube, kb::Cube, 
    v_A::Float64, c_s::Float64; epsilon=0.5*size(ib)[1]^2::Float64, ρ_0=1.0::Float64)::Tuple{Cube,Cube,Cube}
    
    """
    decomposition_A(iv::Cube, jv::Cube, kv::Cube, ib::Cube, jb::Cube, kb::Cube, v_A::Float64, c_s::Float64; epsilon=0.5*size(ib)[1]^2::Float64, ρ_0=1.0::Float64) -> Tuple{Cube,Cube,Cube}

    Decomposes the input velocity and magnetic field cubes into their respective components.

    # Arguments
    - `iv::Cube`: Input velocity component in the i direction.
    - `jv::Cube`: Input velocity component in the j direction.
    - `kv::Cube`: Input velocity component in the k direction.
    - `ib::Cube`: Input magnetic field component in the i direction.
    - `jb::Cube`: Input magnetic field component in the j direction.
    - `kb::Cube`: Input magnetic field component in the k direction.
    - `v_A::Float64`: Alfvén velocity.
    - `c_s::Float64`: Sound speed.
    - `epsilon::Float64`: Small value to avoid division by zero (default: 0.5*size(ib)[1]^2).
    - `ρ_0::Float64`: Initial density (default: 1.0).

    # Returns
    - `Tuple{Cube,Cube,Cube}`: Tuple containing the decomposed magnetic field components (ib_o, jb_o, kb_o).
    """
    
    nx,ny,nz = size(ib)
    β = 2 * c_s^2 / v_A^2
    α = β / 2

    iv_f = fftshift(fft(iv))
    jv_f = fftshift(fft(jv))
    kv_f = fftshift(fft(kv))

    ibf_o = zeros(typeof(1.0.+im), size(iv_f))
    jbf_o = zeros(typeof(1.0.+im), size(jv_f))                                                                                       
    kbf_o = zeros(typeof(1.0.+im), size(kv_f))

    Bhat = [mean(ib), mean(jb), mean(kb)]

    for k in 1:nz, j in 1:ny, i in 1:nx                                                                                                      
        
        kx = i - div(nx,2)-1
        ky = j - div(ny,2)-1
        kz = k - div(nz,2)-1
        kk = [kx,ky,kz]
        
        if (sum(kk.^2.0) > 0)
            khat = normalize(kk)
            kdotB = dot(khat, normalize(Bhat))
            
            D = (1 + α)^2 - (4 * α * kdotB^2)
                                                                                                                                                 
            # slow mode
            prefactor1 = 1-sqrt(D)+α
            prefactor2 = 1+sqrt(D)-α

            # fast mode                                                                                                                          
            prefactor3 = 1-sqrt(D)-α
            prefactor4 = 1+sqrt(D)+α

            if (1.0 .- abs.(kdotB) > epsilon)

                if (mode=="A")
                    ζ_mode = ucross3(kk,Bhat)
                elseif (mode=="C")
                    zeta_A = ucross3(kk,Bhat)
                    zeta_o = ucross3(kk,zeta_A)
                elseif (mode=="P")
                    zeta_o = n3(kk)

                elseif (mode=="S")                                                                                                                      
                    zeta_ll = n3(Bhat)
                    zeta_pp = n3(kk.-cdot3(kk,zeta_ll).*zeta_ll)
                    zeta_x  = -prefactor2.*zeta_ll.+prefactor1.*zeta_pp
                    if (m3(zeta_x)>epsilon)
                        zeta_o  = n3(zeta_x)                                                                                                             
                    else
                        zeta_o = [0,0,0]
                    end                                                                                                                              
                elseif (mode=="F")
                    zeta_ll = n3(Bhat)
                    zeta_pp = n3(kk.-cdot3(kk,zeta_ll).*zeta_ll)
                    zeta_x  = -prefactor3.*zeta_ll.+prefactor4.*zeta_pp
                    if (m3(zeta_x)>epsilon)
                        zeta_o  = n3(zeta_x)
                    else
                        zeta_o = [0,0,0]
                    end
                end                                                                                                                                  
                # assign a vector at (i,j,k)
                vf = [iv_f[i,j,k], jv_f[i,j,k], kv_f[i,j,k]]
                # do a dot product
                vfdotC = dot(vf, ζ_mode)
                
                # velocity vector
                vf_o = vfdotC .* ζ_mode
                
                if (mode == "A" || mode == "S" || mode == "F")
                    c_w = wavespeed(Bhat, khat, v_A, c_s)
                end
                
                # k x (v x B)/c_w = √ρ*v_A/c_w * (khat x (v x Bhat))
                
                if (c_w > 0)
                    ibo, jbo, kbo = double_cross(khat..., vf_o..., Bhat...) / c_w * v_A * sqrt(ρ_0)
                else
                    # mute nonpropagating modes atm.
                    ibo, jbo, kbo = 0, 0, 0
                end
                ibf_o[i,j,k] = ibo
                jbf_o[i,j,k] = jbo
                kbf_o[i,j,k] = kbo
                # 
            end                                                                                                                              
        end
    end
    ib_o = real(ifft(ifftshift(ibf_o)))
    jb_o = real(ifft(ifftshift(jbf_o)))
    kb_o = real(ifft(ifftshift(kbf_o)))
    
    return ib_o, jb_o, kb_o
end
