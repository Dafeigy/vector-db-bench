use std::arch::x86_64::*;

#[inline]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();

        let va0 = _mm512_loadu_ps(a.as_ptr());
        let vb0 = _mm512_loadu_ps(b.as_ptr());
        let diff0 = _mm512_sub_ps(va0, vb0);
        sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);

        let va1 = _mm512_loadu_ps(a.as_ptr().add(16));
        let vb1 = _mm512_loadu_ps(b.as_ptr().add(16));
        let diff1 = _mm512_sub_ps(va1, vb1);
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);

        let va2 = _mm512_loadu_ps(a.as_ptr().add(32));
        let vb2 = _mm512_loadu_ps(b.as_ptr().add(32));
        let diff2 = _mm512_sub_ps(va2, vb2);
        sum0 = _mm512_fmadd_ps(diff2, diff2, sum0);

        let va3 = _mm512_loadu_ps(a.as_ptr().add(48));
        let vb3 = _mm512_loadu_ps(b.as_ptr().add(48));
        let diff3 = _mm512_sub_ps(va3, vb3);
        sum1 = _mm512_fmadd_ps(diff3, diff3, sum1);

        let va4 = _mm512_loadu_ps(a.as_ptr().add(64));
        let vb4 = _mm512_loadu_ps(b.as_ptr().add(64));
        let diff4 = _mm512_sub_ps(va4, vb4);
        sum0 = _mm512_fmadd_ps(diff4, diff4, sum0);

        let va5 = _mm512_loadu_ps(a.as_ptr().add(80));
        let vb5 = _mm512_loadu_ps(b.as_ptr().add(80));
        let diff5 = _mm512_sub_ps(va5, vb5);
        sum1 = _mm512_fmadd_ps(diff5, diff5, sum1);

        let va6 = _mm512_loadu_ps(a.as_ptr().add(96));
        let vb6 = _mm512_loadu_ps(b.as_ptr().add(96));
        let diff6 = _mm512_sub_ps(va6, vb6);
        sum0 = _mm512_fmadd_ps(diff6, diff6, sum0);

        let va7 = _mm512_loadu_ps(a.as_ptr().add(112));
        let vb7 = _mm512_loadu_ps(b.as_ptr().add(112));
        let diff7 = _mm512_sub_ps(va7, vb7);
        sum1 = _mm512_fmadd_ps(diff7, diff7, sum1);

        sum0 = _mm512_add_ps(sum0, sum1);

        _mm512_reduce_add_ps(sum0)
    }
}
