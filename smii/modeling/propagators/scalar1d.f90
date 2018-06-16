! Based on version found in PySIT
! https://github.com/pysit/pysit/blob/master/pysit/solvers/constant_density_acoustic/time/scalar/constant_density_acoustic_time_scalar_1D_4.h

module scalar1d

        implicit none

contains

        subroutine step(f1, f2, phix1, phix2, sigmax, model2_dt2,      &
                        dx, dt, sources_amp, sources_loc, num_steps,   &
                        pml_width, step_ratio)

                real, intent (in out), dimension (:, :) :: f1
                real, intent (in out), dimension (:, :) :: f2
                real, intent (in out), dimension (:, :) :: phix1
                real, intent (in out), dimension (:, :) :: phix2
                real, intent (in), dimension (:) :: sigmax
                real, intent (in), dimension (:) :: model2_dt2
                real, intent (in) :: dx
                real, intent (in) :: dt
                real, intent (in), dimension (:, :, :) :: sources_amp
                integer, intent (in), dimension (:, :) :: sources_loc
                integer, intent (in) :: num_steps
                integer, intent (in) :: pml_width
                integer, intent (in) :: step_ratio

                integer :: inner_step_idx
                logical :: even
                real, dimension(2) :: fd_coeff1
                real, dimension(3) :: fd_coeff2

                fd_coeff1 = (/                                         &
                        8.0 / 12 / dx,                                 &
                        -1.0 / 12 / dx                                 &
                        /)

                fd_coeff2 = (/                                         &
                        -30.0 / 12 / dx**2 / 2,                        &
                        16.0 / 12 / dx**2,                             &
                        -1.0 / 12 / dx**2                              &
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

                real, intent (in out), dimension (:, :) :: f
                real, intent (in out), dimension (:, :) :: fp
                real, intent (in), dimension (:, :) :: phix
                real, intent (in out), dimension (:, :) :: phixp
                real, intent (in), dimension (:) :: sigmax
                real, intent (in), dimension (:) :: model2_dt2
                real, intent (in) :: dt
                real, intent (in), dimension (:, :, :) :: sources_amp
                integer, intent (in), dimension (:, :) :: sources_loc
                integer, intent (in) :: inner_step_idx
                real, intent (in), dimension (2) :: fd_coeff1
                real, intent (in), dimension (3) :: fd_coeff2
                integer, intent (in) :: pml_width
                integer, intent (in) :: step_ratio

                integer :: x
                integer :: s
                integer :: nx
                integer :: lpml1
                integer :: lpml2
                integer :: rpml1
                integer :: rpml2
                integer :: ix1
                integer :: ix2

                nx = size(f, dim=1)

                lpml1 = 3
                lpml2 = lpml1 + pml_width - 1

                rpml2 = nx - 2 !2
                rpml1 = rpml2 - pml_width + 1 !1

                ix1 = lpml2 + 1
                ix2 = rpml1 - 1

                !$omp parallel do default(none)                        &
                !$omp& shared(fd_coeff1, fd_coeff2, f, fp)             &
                !$omp& shared(model2_dt2)                              &
                !$omp& shared(lpml1, rpml2, phix)                      &
                !$omp& shared(phixp, sigmax, pml_width)                &
                !$omp& shared(dt, ix1, ix2) collapse(2)
                do s = 1, ubound(fp, dim=2)
                do x = lpml1, rpml2
                if ((x < ix1) .or. (x > ix2)) then
                        call fd_pml(f, fp, phix, phixp, sigmax,        &
                                model2_dt2, dt, fd_coeff1, fd_coeff2,  &
                                x, s)
                else
                        call fd_inner(f, fp, phix, model2_dt2,         &
                                fd_coeff1, fd_coeff2, x, s)
                end if
                end do
                end do

                !$omp parallel do default(none)                        &
                !$omp& shared(fp, model2_dt2, sources_amp, sources_loc)&
                !$omp& shared(inner_step_idx, step_ratio)
                do s = 1, ubound(fp, dim=2)
                call add_sources(fp, model2_dt2, sources_amp,          &
                        sources_loc, inner_step_idx,                   &
                        step_ratio, s)
                end do

        end subroutine one_step


        subroutine add_sources(fp, model2_dt2, sources_amp,            &
                        sources_loc, step_idx, step_ratio, s)

                real, intent (in out), dimension (:, :) :: fp
                real, intent (in), dimension (:) :: model2_dt2
                real, intent (in), dimension (:, :, :) :: sources_amp
                integer, intent (in), dimension (:, :) :: sources_loc
                integer, intent (in) :: step_idx
                integer, intent (in) :: step_ratio
                integer, intent (in) :: s

                integer :: sx
                integer :: i
                integer :: num_sources

                num_sources = size(sources_amp, dim=2)

                do i = 1, num_sources
                sx = sources_loc(i, s) + 1
                fp(sx, s) = fp(sx, s) + model2_dt2(sx)                 &
                        * sources_amp((step_idx-1)/step_ratio+1, i, s)
                end do

        end subroutine add_sources


        subroutine fd_inner(f, fp, phix, model2_dt2,                   &
                        fd_coeff1, fd_coeff2, x, s)

                real, intent (in), dimension (:, :) :: f
                real, intent (in out), dimension (:, :) :: fp
                real, intent (in), dimension (:, :) :: phix
                real, intent (in), dimension (:) :: model2_dt2
                real, intent (in), dimension (2) :: fd_coeff1
                real, intent (in), dimension (3) :: fd_coeff2
                integer, intent (in) :: x
                integer, intent (in) :: s

                real :: f_xx
                real :: phix_x

                f_xx = d2fdx2(f, x, s, fd_coeff2)
                phix_x = dfdx(phix, x, s, fd_coeff1)

                fp(x, s) = model2_dt2(x) * (f_xx + phix_x) +           &
                        2 * f(x, s) - fp(x, s)

        end subroutine fd_inner


        subroutine fd_pml(f, fp, phix, phixp, sigmax, model2_dt2, dt,  &
                        fd_coeff1, fd_coeff2, x, s)

                real, intent (in), dimension (:, :) :: f
                real, intent (in out), dimension (:, :) :: fp
                real, intent (in), dimension (:, :) :: phix
                real, intent (in out), dimension (:, :) :: phixp
                real, intent (in), dimension (:) :: sigmax
                real, intent (in), dimension (:) :: model2_dt2
                real, intent (in) :: dt
                real, intent (in), dimension (2) :: fd_coeff1
                real, intent (in), dimension (3) :: fd_coeff2
                integer, intent (in) :: x
                integer, intent (in) :: s

                real :: f_xx
                real :: f_x
                real :: phix_x
                real :: factor

                f_xx = d2fdx2(f, x, s, fd_coeff2)
                f_x = dfdx(f, x, s, fd_coeff1)
                phix_x = dfdx(phix, x, s, fd_coeff1)
                factor = 1 / (1 + dt * sigmax(x) / 2)

                fp(x, s) = factor * (model2_dt2(x) *                   &
                        (f_xx + phix_x) +                              &
                        dt * sigmax(x) * fp(x, s) / 2 +                &
                        (2 * f(x, s) - fp(x, s)))

                phixp(x, s) = phix(x, s) -                             &
                        dt * sigmax(x) * (f_x + phix(x, s))

        end subroutine fd_pml


        pure function dfdx(f, x, s, fd_coeff)

                real, intent (in), dimension (:, :) :: f
                integer, intent (in) :: x
                integer, intent (in) :: s
                real, intent (in), dimension (2) :: fd_coeff

                real :: dfdx
                integer :: i

                dfdx = 0.0

                do i = 1, 2
                dfdx = dfdx + fd_coeff(i) * (f(x + i, s) - f(x - i, s))
                end do

        end function dfdx


        pure function d2fdx2(f, x, s, fd_coeff)

                real, intent (in), dimension (:, :) :: f
                integer, intent (in) :: x
                integer, intent (in) :: s
                real, intent (in), dimension (3) :: fd_coeff

                real :: d2fdx2
                integer :: i

                d2fdx2 = 0.0

                do i = 0, 2
                d2fdx2 = d2fdx2 +                                      &
                        fd_coeff(i + 1) * (f(x + i, s) + f(x - i, s))
                end do

        end function d2fdx2


end module scalar1d
