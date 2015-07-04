! %fortran

        Subroutine Rescale(a,b,n1,n2)
        Implicit none
        Integer n1,n2,i,j
        Real*8 a(n1,n2), b, pi
        pi = 4.*atan(1.)
Cf2py intent(in,out,overwrite) a
        do i = 1,n1
           do j=1,n2
             a(i,j)=b*a(i,j)
           end do
        end do
        end