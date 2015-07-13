! %fortran
! 
        Subroutine polarize(im0,at3,r,imp,n1,n2,rMax,tMax)
        Implicit none
        Integer at3(n1,n2),r(n1,n2),n1,n2,i,j,rMax,tmax
        Real*8 im0(n1,n2),imp(tMax,rMax)


Cf2py intent(in,out,overwrite) imp
! ,im0,at3,r
        ! n1, n2 are preserved coreectly -> n1=619, n2=487  
        ! write (*,*) 'n1: ', n1, ' n2: ', n2
        do i = 1,n1
            ! x=i-colC      
            do j=1,n2

            ! have to add the +1 because at3 and r are in index[0] base, but fortran is using index[1]
            ! since at3 and r are coord maps generated in numpy, they are indices in imp (index base 1 here) that take values
            ! from im0 (index base 0, since we constructed it in numpy). So we have to add 1 to get them to line up in fortran.
            imp(at3(i,j)+1,r(i,j)+1)=imp(at3(i,j)+1,r(i,j)+1)+im0(i,j)
            end do
        end do
        end


! compile: f2py -c polarize.f --f90flags="-ffixed-line-length-132" -m polarize
! the f90flags probably aren't needed, and probably don't cure what I was trying to cure. Try it without.
