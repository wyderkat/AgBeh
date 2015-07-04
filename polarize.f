! %fortran
! 
        Subroutine polarize(im0,at3,r,imp,n1,n2,rMax,tMax)
        Implicit none
        Integer at3(n1,n2),r(n1,n2),n1,n2,i,j,rMax,tmax
        Real*8 im0(n1,n2),imp(tMax,rMax)
        ! pi = 4.*atan(1.)
! Cf2py intent(in) im0,at3,r
Cf2py intent(in,out,overwrite) imp
! ,im0,at3,r
        ! n1, n2 are preserved coreectly -> n1=619, n2=487  
        ! write (*,*) 'n1: ', n1, ' n2: ', n2
        do i = 1,n1
            ! x=i-colC      
            do j=1,n2
                ! y=j-rowC

                ! a(i,j)=b*a(i,j)
            ! have to add the +1 because at3 and r are in index[0] base, but fortran is using index[1]
            ! since at3 and r are a coord maps generated in numpy, they are indices in imp that take values
            ! from im0. So we have to add 1 to get them to line up in fortran.
            imp(at3(i,j)+1,r(i,j)+1)=imp(at3(i,j)+1,r(i,j)+1)+im0(i,j)
            end do
        end do
        end


            !     for x in range(xM):
            ! vP.SetX(x-colCenter)
            ! for y in range(yM):
                
            !     vP.SetY(y-rowCenter)
            !     p=vP.Phi()*pSize/(2*pi)
            !     r=vP.Mod()
            !     # p=1327./(2*pi)*p
            !     # imPolarHist.Fill(r,p,im0[y,x])
            !     # print p,r
            !     imPolar[round(p),round(r)]+=im0[y,x]