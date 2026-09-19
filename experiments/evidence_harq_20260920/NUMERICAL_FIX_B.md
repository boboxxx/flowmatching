# Candidate B numerical failure and bounded transmitter fix

The unbounded enhancement encoder exceeded binary16 packet-scale representability
after step 1950. The receiver-side validation correctly stopped training instead of
allowing inf/NaN or substituting an uncharged clean scale. This is a failed execution,
not a completed negative algorithm result. The last valid checkpoint and log are kept.

B-fixed uses tanh(layer_norm(E1_raw(x,z0))) before packetization. This removes an
unbounded transmitter scale degree of freedom and constrains enhancement values to
[-1,1]. Normalization statistics stay at the transmitter; actual quantized packet scale
is still sent and charged. No receiver information or gate tolerance changes. The
training budget/seed/splits are exactly B's; it starts fresh, not from the failed weights.
This is an implementation repair recorded separately from A and B, not a passed gate.

The layer-normalized target, if a posterior is trained, must be the NEW frozen codec's
evidence. Unbounded-candidate posterior weights or statistics would be incompatible.
