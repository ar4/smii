! Based on version found in PySIT
! https://github.com/pysit/pysit/blob/master/pysit/solvers/constant_density_acoustic/time/scalar/constant_density_acoustic_time_scalar_1D_4.h

module scalar1d16

        implicit none

contains

        subroutine step(f1, f2, phix1, phix2, sigmax, model2_dt2,      &
                        dx, dt, sources_amp, sources_loc, num_steps,   &
                        pml_width, step_ratio)

                real, intent (in out), dimension (:) :: f1
                real, intent (in out), dimension (:) :: f2
                real, intent (in out), dimension (:) :: phix1
                real, intent (in out), dimension (:) :: phix2
                real, intent (in), dimension (:) :: sigmax
                real, intent (in), dimension (:) :: model2_dt2
                real, intent (in) :: dx
                real, intent (in) :: dt
                real, intent (in), dimension (:, :) :: sources_amp
                integer, intent (in), dimension (:) :: sources_loc
                integer, intent (in) :: num_steps
                integer, intent (in) :: pml_width
                integer, intent (in) :: step_ratio

                integer :: inner_step_idx
                logical :: even
                real, dimension(6) :: fd_coeff1
                real, dimension(9) :: fd_coeff2

                fd_coeff1 = (/                                         &
                        23760.0 / 27720 / dx,                          &
                        -7425.0 / 27720 / dx,                          &
                        2200.0 / 27720 / dx,                           &
                        -495.0 / 27720 / dx,                           &
                        72.0 / 27720 / dx,                             &
                        -5.0 / 27720 / dx                              &
                        /)

                fd_coeff2 = (/                                         &
                        -924708642.0 / 302702400 / (dx * dx) / 2,      &
                        538137600.0 / 302702400 / (dx * dx),           &
                        -94174080.0 / 302702400 / (dx * dx),           &
                        22830080.0 / 302702400 / (dx * dx),            &
                        -5350800.0 / 302702400 / (dx * dx),            &
                        1053696.0 / 302702400 / (dx * dx),             &
                        -156800.0 / 302702400 / (dx * dx),             &
                        15360.0 / 302702400 / (dx * dx),               &
                        -735.0 / 302702400 / (dx * dx)                 &
                        /)

                do inner_step_idx = 1, num_steps * step_ratio
                even = (mod (inner_step_idx, 2) == 0)
                if (even) then
                        call one_step(f2, f1, phix2, phix1, sigmax,    &
                                model2_dt2, dt, sources_amp,           &
                                sources_loc, inner_step_idx,           &
                                fd_coeff1, fd_coeff2, pml_width,       &
                                step_ratio)
                else
                        call one_step(f1, f2, phix1, phix2, sigmax,    &
                                model2_dt2, dt, sources_amp,           &
                                sources_loc, inner_step_idx,           &
                                fd_coeff1, fd_coeff2, pml_width,       &
                                step_ratio)
                end if
                end do

        end subroutine step


        subroutine one_step(f, fp, phix, phixp, sigmax, model2_dt2,    &
                        dt, sources_amp, sources_loc, inner_step_idx,  &
                        fd_coeff1, fd_coeff2, pml_width, step_ratio)

                real, intent (in out), dimension (:) :: f
                real, intent (in out), dimension (:) :: fp
                real, intent (in), dimension (:) :: phix
                real, intent (in out), dimension (:) :: phixp
                real, intent (in), dimension (:) :: sigmax
                real, intent (in), dimension (:) :: model2_dt2
                real, intent (in) :: dt
                real, intent (in), dimension (:, :) :: sources_amp
                integer, intent (in), dimension (:) :: sources_loc
                integer, intent (in) :: inner_step_idx
                real, intent (in), dimension (6) :: fd_coeff1
                real, intent (in), dimension (9) :: fd_coeff2
                integer, intent (in) :: pml_width
                integer, intent (in) :: step_ratio

                integer :: x
                integer :: nx
                integer :: lpml1
                integer :: lpml2
                integer :: rpml1
                integer :: rpml2
                integer :: ix1
                integer :: ix2

                nx = size(f)

                lpml1 = 9
                lpml2 = lpml1 + pml_width - 1

                rpml2 = nx - 8 !2
                rpml1 = rpml2 - pml_width + 1 !1

                ix1 = lpml2 + 1
                ix2 = rpml1 - 1

                !$omp parallel do default(none)                        &
                !$omp& shared(nx, fd_coeff1, fd_coeff2, f, fp)         &
                !$omp& shared(model2_dt2)                              &
                !$omp& shared(lpml1, lpml2, rpml1, rpml2, phix)        &
                !$omp& shared(phixp, sigmax, pml_width)                &
                !$omp& shared(dt, ix1, ix2)
                do x = lpml1, rpml2
                if ((x < ix1) .or. (x > ix2)) then
                        call fd_pml(f, fp, phix, phixp, sigmax,        &
                                model2_dt2, dt, fd_coeff1, fd_coeff2, x)
                else
                        call fd_inner(f, fp, phix, model2_dt2,         &
                                fd_coeff1, fd_coeff2, x)
                end if
                end do

                !if (mod(inner_step_idx, step_ratio) == 0) then
                !        call add_sources(fp, model2_dt2, sources_amp,      &
                !                sources_loc, inner_step_idx / step_ratio,&
                !                step_ratio)
                !end if

                call add_sources(fp, model2_dt2, sources_amp,          &
                        sources_loc, inner_step_idx,                   &
                        step_ratio)

        end subroutine one_step


        subroutine add_sources(fp, model2_dt2, sources_amp,            &
                        sources_loc, step_idx, step_ratio)

                real, intent (in out), dimension (:) :: fp
                real, intent (in), dimension (:) :: model2_dt2
                real, intent (in), dimension (:, :) :: sources_amp
                integer, intent (in), dimension (:) :: sources_loc
                integer, intent (in) :: step_idx
                integer, intent (in) :: step_ratio

                integer :: sx
                integer :: i
                integer :: num_sources

                num_sources = size(sources_amp, dim=2)

                do i = 1, num_sources
                sx = sources_loc(i) + 1
                fp(sx) = fp(sx) + model2_dt2(sx)                       &
                        * sources_amp((step_idx-1)/step_ratio+1, i)
                end do

        end subroutine add_sources


        subroutine fd_inner(f, fp, phix, model2_dt2,                   &
                        fd_coeff1, fd_coeff2, x)

                real, intent (in), dimension (:) :: f
                real, intent (in out), dimension (:) :: fp
                real, intent (in), dimension (:) :: phix
                real, intent (in), dimension (:) :: model2_dt2
                real, intent (in), dimension (6) :: fd_coeff1
                real, intent (in), dimension (9) :: fd_coeff2
                integer, intent (in) :: x

                real :: f_xx
                real :: phix_x

                f_xx = d2fdx2(f, x, fd_coeff2)
                phix_x = dfdx(phix, x, fd_coeff1)

                fp(x) = model2_dt2(x) * (f_xx + phix_x) +              &
                        2 * f(x) - fp(x)

        end subroutine fd_inner


        subroutine fd_pml(f, fp, phix, phixp, sigmax, model2_dt2, dt,  &
                        fd_coeff1, fd_coeff2, x)

                real, intent (in), dimension (:) :: f
                real, intent (in out), dimension (:) :: fp
                real, intent (in), dimension (:) :: phix
                real, intent (in out), dimension (:) :: phixp
                real, intent (in), dimension (:) :: sigmax
                real, intent (in), dimension (:) :: model2_dt2
                real, intent (in) :: dt
                real, intent (in), dimension (6) :: fd_coeff1
                real, intent (in), dimension (9) :: fd_coeff2
                integer, intent (in) :: x

                real :: f_xx
                real :: f_x
                real :: phix_x
                real :: factor

                f_xx = d2fdx2(f, x, fd_coeff2)
                f_x = dfdx(f, x, fd_coeff1)
                phix_x = dfdx(phix, x, fd_coeff1)
                factor = 1 / (1 + dt * sigmax(x) / 2)

                fp(x) = factor * (model2_dt2(x) *                      &
                        (f_xx + phix_x) +                              &
                        dt * sigmax(x) * fp(x) / 2 +                   &
                        (2 * f(x) - fp(x)))

                phixp(x) = phix(x) - dt * sigmax(x) * (f_x + phix(x))

        end subroutine fd_pml


        pure function dfdx(f, x, fd_coeff)

                real, intent (in), dimension (:) :: f
                integer, intent (in) :: x
                real, intent (in), dimension (6) :: fd_coeff

                real :: dfdx
                integer :: i

                dfdx = 0.0

                do i = 1, 6
                dfdx = dfdx + fd_coeff(i) * (f(x + i) - f(x - i))
                end do

        end function dfdx


        pure function d2fdx2(f, x, fd_coeff)

                real, intent (in), dimension (:) :: f
                integer, intent (in) :: x
                real, intent (in), dimension (9) :: fd_coeff

                real :: d2fdx2
                integer :: i

                d2fdx2 = 0.0

                do i = 0, 8
                d2fdx2 = d2fdx2 +                                      &
                        fd_coeff(i + 1) * (f(x + i) + f(x - i))
                end do

        end function d2fdx2


end module scalar1d16
