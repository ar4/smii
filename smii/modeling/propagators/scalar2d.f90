! Based on version found in PySIT
! https://github.com/pysit/pysit/blob/master/pysit/solvers/constant_density_acoustic/time/scalar/constant_density_acoustic_time_scalar_2D_4.h

module scalar2d

        implicit none

contains

        subroutine step(f1, f2, phix1, phix2, phiy1, phiy2, sigmax,    &
                        sigmay, model2_dt2, nxi, dx, dt,               &
                        sources, sources_x, num_steps,                 &
                        pml_width, step_ratio)

                real, intent (in out), dimension (:, :) :: f1
                real, intent (in out), dimension (:, :) :: f2
                real, intent (in out), dimension (:, :) :: phix1
                real, intent (in out), dimension (:, :) :: phix2
                real, intent (in out), dimension (:, :) :: phiy1
                real, intent (in out), dimension (:, :) :: phiy2
                real, intent (in), dimension (:) :: sigmax
                real, intent (in), dimension (:) :: sigmay
                real, intent (in), dimension (:, :) :: model2_dt2
                integer, intent (in) :: nxi
                real, intent (in) :: dx
                real, intent (in) :: dt
                real, intent (in), dimension (:, :) :: sources
                integer, intent (in), dimension (:, :) :: sources_x
                integer, intent (in) :: num_steps
                integer, intent (in) :: pml_width
                integer, intent (in) :: step_ratio

                integer :: step_idx
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


                do step_idx = 1, num_steps * step_ratio
                even = (mod (step_idx, 2) == 0)
                if (even) then
                        call one_step(f2, f1, phix2, phix1, phiy2,     &
                                phiy1, sigmax, sigmay,                 &
                                model2_dt2, nxi,                       &
                                dt, sources, sources_x,                &
                                step_idx, fd_coeff1, fd_coeff2,        &
                                pml_width, step_ratio)
                else
                        call one_step(f1, f2, phix1, phix2, phiy1,     &
                                phiy2, sigmax, sigmay,                 &
                                model2_dt2, nxi,                       &
                                dt, sources, sources_x,                &
                                step_idx, fd_coeff1, fd_coeff2,        &
                                pml_width, step_ratio)
                end if
                end do

        end subroutine step


        subroutine one_step(f, fp, phix, phixp, phiy, phiyp, sigmax,   &
                        sigmay, model2_dt2, nxi, dt,                   &
                        sources, sources_x, step_idx,                  &
                        fd_coeff1, fd_coeff2, pml_width, step_ratio)

                real, intent (in), dimension (:, :) :: f
                real, intent (in out), dimension (:, :) :: fp
                real, intent (in out), dimension (:, :) :: phix
                real, intent (in out), dimension (:, :) :: phixp
                real, intent (in out), dimension (:, :) :: phiy
                real, intent (in out), dimension (:, :) :: phiyp
                real, intent (in), dimension (:) :: sigmax
                real, intent (in), dimension (:) :: sigmay
                real, intent (in), dimension (:, :) :: model2_dt2
                integer, intent (in) :: nxi
                real, intent (in) :: dt
                real, intent (in), dimension (:, :) :: sources
                integer, intent (in), dimension (:, :) :: sources_x
                integer, intent (in) :: step_idx
                real, intent (in), dimension (2) :: fd_coeff1
                real, intent (in), dimension (3) :: fd_coeff2
                integer, intent (in) :: pml_width
                integer, intent (in) :: step_ratio

                integer :: x
                integer :: y
                integer :: ny
                integer :: tpml1
                integer :: tpml2
                integer :: bpml1
                integer :: bpml2
                integer :: lpml1
                integer :: lpml2
                integer :: rpml1
                integer :: rpml2
                integer :: ix1
                integer :: ix2
                !integer :: iy1
                !integer :: iy2

                ny = size(f, dim=2)

                tpml1 = 5
                tpml2 = tpml1 + pml_width - 3

                bpml2 = ny - 4 !2
                bpml1 = bpml2 - pml_width + 3 !1

                lpml1 = 5
                lpml2 = lpml1 + pml_width - 3

                rpml1 = lpml2 + nxi + 3
                rpml2 = rpml1 + pml_width - 3

                ix1 = lpml2 + 1
                ix2 = rpml1 - 1
                !iy1 = tpml2 + 1
                !iy2 = bpml2 - 1

                !$omp parallel do default(none) private(x)             &
                !$omp& shared(fd_coeff1, fd_coeff2, f, fp)             &
                !$omp& shared(model2_dt2, tpml1, tpml2, bpml1, bpml2)  &
                !$omp& shared(lpml1, lpml2, rpml1, rpml2, phix, phiy)  &
                !$omp& shared(phixp, phiyp, sigmax, sigmay, pml_width) &
                !$omp& shared(dt, ix1, ix2)
                do y = tpml1, bpml2
                if ((y <= tpml2) .or. (y >= bpml1)) then

                        do x = lpml1, rpml2
                        call fd_pml(f, fp, phix, phixp, phiy, phiyp,   &
                                sigmax, sigmay, model2_dt2, dt,        &
                                fd_coeff1, fd_coeff2, x, y)
                        end do

                else

                        do x = lpml1, lpml2
                        call fd_pml(f, fp, phix, phixp, phiy, phiyp,   &
                                sigmax, sigmay, model2_dt2, dt,        &
                                fd_coeff1, fd_coeff2, x, y)
                        end do

                        do x = ix1, ix2
                        call fd_inner(f, fp,                           &
                                model2_dt2,                            &
                                fd_coeff2, x, y)
                        end do

                        do x = rpml1, rpml2
                        call fd_pml(f, fp, phix, phixp, phiy, phiyp,   &
                                sigmax, sigmay, model2_dt2, dt,        &
                                fd_coeff1, fd_coeff2, x, y)
                        end do

                end if
                end do

                call add_sources(fp, model2_dt2, sources,              &
                        sources_x, step_idx, step_ratio)

        end subroutine one_step


        subroutine add_sources(fp, model2_dt2, sources,                &
                        sources_x, step_idx, step_ratio)

                real, intent (in out), dimension (:, :) :: fp
                real, intent (in), dimension (:, :) :: model2_dt2
                real, intent (in), dimension (:, :) :: sources
                integer, intent (in), dimension (:, :) :: sources_x
                integer, intent (in) :: step_idx
                integer, intent (in) :: step_ratio

                integer :: sx
                integer :: sy
                integer :: i
                integer :: num_sources

                num_sources = size(sources, dim=2)

                do i = 1, num_sources
                sx = sources_x(2, i) + 1
                sy = sources_x(1, i) + 1
                fp(sx, sy) = fp(sx, sy) +                              &
                        model2_dt2(sx, sy) *                           &
                        sources((step_idx-1)/step_ratio+1, i)
                end do

        end subroutine add_sources


        subroutine fd_inner(f, fp, model2_dt2,                         &
                        fd_coeff2, x, y)

                real, intent (in), dimension (:, :) :: f
                real, intent (in out), dimension (:, :) :: fp
                real, intent (in), dimension (:, :) :: model2_dt2
                real, intent (in), dimension (3) :: fd_coeff2
                integer, intent (in) :: x
                integer, intent (in) :: y

                real :: lap

                lap = fd_coeff2(1) *                                   &
                        (f(x + 0, y) + f(x - 0, y) +                   &
                        f(x, y + 0) + f(x, y - 0)) +                   &
                        fd_coeff2(2) *                                 &
                        (f(x + 1, y) + f(x - 1, y) +                   &
                        f(x, y + 1) + f(x, y - 1)) +                   &
                        fd_coeff2(3) *                                 &
                        (f(x + 2, y) + f(x - 2, y) +                   &
                        f(x, y + 2) + f(x, y - 2))

                fp(x, y) = model2_dt2(x, y) *                          &
                        lap +                                          &
                        2 * f(x, y) - fp(x, y)

        end subroutine fd_inner


        subroutine fd_pml(f, fp, phix, phixp, phiy, phiyp, sigmax,     &
                        sigmay, model2_dt2, dt, fd_coeff1,             &
                        fd_coeff2, x, y)

                real, intent (in), dimension (:, :) :: f
                real, intent (in out), dimension (:, :) :: fp
                real, intent (in), dimension (:, :) :: phix
                real, intent (in out), dimension (:, :) :: phixp
                real, intent (in), dimension (:, :) :: phiy
                real, intent (in out), dimension (:, :) :: phiyp
                real, intent (in), dimension (:) :: sigmax
                real, intent (in), dimension (:) :: sigmay
                real, intent (in) :: dt
                real, intent (in), dimension (:, :) :: model2_dt2
                real, intent (in), dimension (2) :: fd_coeff1
                real, intent (in), dimension (3) :: fd_coeff2
                integer, intent (in) :: x
                integer, intent (in) :: y

                real :: lap
                real :: f_x
                real :: f_y
                real :: phix_x
                real :: phiy_y
                real :: sigma_sum
                real :: factor

                lap = fd_coeff2(1) *                                   &
                        (f(x + 0, y) + f(x - 0, y) +                   &
                        f(x, y + 0) + f(x, y - 0)) +                   &
                        fd_coeff2(2) *                                 &
                        (f(x + 1, y) + f(x - 1, y) +                   &
                        f(x, y + 1) + f(x, y - 1)) +                   &
                        fd_coeff2(3) *                                 &
                        (f(x + 2, y) + f(x - 2, y) +                   &
                        f(x, y + 2) + f(x, y - 2))

                f_x = fd_coeff1(1) *                                   &
                        (f(x + 1, y) - f(x - 1, y)) +                  &
                        fd_coeff1(2) *                                 &
                        (f(x + 2, y) - f(x - 2, y))

                f_y = fd_coeff1(1) *                                   &
                        (f(x, y + 1) - f(x, y - 1)) +                  &
                        fd_coeff1(2) *                                 &
                        (f(x, y + 2) - f(x, y - 2))

                phix_x = fd_coeff1(1) *                                &
                        (phix(x + 1, y) - phix(x - 1, y)) +            &
                        fd_coeff1(2) *                                 &
                        (phix(x + 2, y) - phix(x - 2, y))

                phiy_y = fd_coeff1(1) *                                &
                        (phiy(x, y + 1) - phiy(x, y - 1)) +            &
                        fd_coeff1(2) *                                 &
                        (phiy(x, y + 2) - phiy(x, y - 2))

                sigma_sum = sigmax(x) + sigmay(y)
                factor = 1 / (1 + dt * sigma_sum / 2)

                fp(x, y) = factor * (model2_dt2(x, y) *                &
                        (lap + phix_x + phiy_y) +                      &
                        dt * sigma_sum * fp(x, y) / 2 +                &
                        (2 * f(x, y) - fp(x, y)) -                     &
                        dt**2 * sigmax(x) * sigmay(y) * f(x, y))

                phixp(x, y) = phix(x, y) -                             &
                        dt * sigmax(x) * phix(x, y) -                  &
                        dt * (sigmax(x) - sigmay(y)) * f_x
                phiyp(x, y) = phiy(x, y) -                             &
                        dt * sigmay(y) * phiy(x, y) -                  &
                        dt * (sigmay(y) - sigmax(x)) * f_y

        end subroutine fd_pml

end module scalar2d
