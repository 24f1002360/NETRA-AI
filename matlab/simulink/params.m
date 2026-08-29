% params.m
% ---------------------------------------------------------------------
% All NETRA simulation parameters live here. Never type numbers into
% block dialogs - change them here and re-run, so every result is
% reproducible and sweeps are possible.
%
% MEASURE  = replace with a real number from the team
% PLACEHOLDER = derived assumption, state it openly in the deck
% ---------------------------------------------------------------------

P = struct();

% --- clinic day ------------------------------------------------------
P.day_minutes          = 480;    % 8-hour PHC day
P.arrivals_per_day     = 70;     % PLACEHOLDER: patients presenting to screen
P.mean_interarrival_min = P.day_minutes / P.arrivals_per_day;

% --- capture station -------------------------------------------------
P.n_capture_stations   = 1;      % cameras + operators at this PHC
P.capture_mean_min     = 5.0;    % PLACEHOLDER: register + position + shoot
P.queue_capacity       = 100;

% --- AI inference ----------------------------------------------------
P.inference_mean_min   = 0.5;    % MEASURE: from Kanchan's CPU benchmark
P.model_size_mb        = 10;     % MEASURE: deployed INT8 model footprint

% --- Quality Gate ----------------------------------------------------
P.retake_prob          = 0.10;   % MEASURE: from the Quality Gate evaluation
P.retake_prob_nogate   = 0.30;   % PLACEHOLDER: retake rate without the gate
P.max_retakes          = 2;

% --- district --------------------------------------------------------
P.n_phcs               = 8;      % PHCs feeding one district hospital
P.n_specialists        = 1;      % ophthalmologists at district level
P.referral_frac        = 0.30;   % PLACEHOLDER: fraction needing specialist review
P.review_min_netra     = 0.5;    % specialist review WITH the visual report
P.review_min_manual    = 3.0;    % specialist review WITHOUT the system

% --- sync ------------------------------------------------------------
P.payload_mb           = 2.0;    % MEASURE: from Divyanshu's sync queue
P.bandwidth_kbps       = 256;    % PLACEHOLDER
P.connectivity_hrs     = 4;      % PLACEHOLDER: usable hours per day
