% build_netra_model.m
% ---------------------------------------------------------------------
% Builds the NETRA district SimEvents model programmatically, so you do
% not have to drag and drop blocks.
%
% HOW TO RUN
%   1. Open MATLAB Online, upload this file (and params.m) to MATLAB Drive
%   2. In the Command Window:   build_netra_model
%   3. The model opens. Press Run.
%
% WHAT IT BUILDS
%
%   Entity Generator --> Entity Queue --> Entity Server --> Output Switch
%       ^                                                      |  |
%       |                                                      |  +--> Terminator (passed)
%       +---------------- retake loop -------------------------+
%
%   The retake loop is the point of the model: images that fail the
%   Quality Gate go back to the capture station.
%
% NOTE ON DIALOG PARAMETERS
%   Block parameter names differ slightly between MATLAB releases, so the
%   set_param calls below are wrapped in try/catch. If one is skipped, the
%   script tells you which block to open and set by hand. The wiring is
%   always built correctly - only a couple of dialog values may need a
%   manual touch.
% ---------------------------------------------------------------------

clear; clc;
params;                                   % load P (see params.m)

model = 'NETRA_district';
if bdIsLoaded(model), close_system(model, 0); end
if exist([model '.slx'], 'file'), delete([model '.slx']); end

new_system(model);
open_system(model);
set_param(model, 'StopTime', num2str(P.day_minutes));

L = 'simevents/';                          % SimEvents library

add_block([L 'Entity Generator'],  [model '/Arrivals'],   'Position', [ 60  80 130 130]);
add_block([L 'Entity Queue'],      [model '/WaitingRoom'],'Position', [200  80 260 130]);
add_block([L 'Entity Server'],     [model '/Capture'],    'Position', [330  80 390 130]);
add_block([L 'Entity Output Switch'],[model '/QualityGate'],'Position',[460  80 510 130]);
add_block([L 'Entity Terminator'], [model '/Screened'],   [ 'Position'], [610  60 650 100]);
add_block([L 'Entity Terminator'], [model '/EndOfDay'],   'Position', [610 150 650 190]);

% ---- wiring ---------------------------------------------------------
add_line(model, 'Arrivals/1',    'WaitingRoom/1', 'autorouting','on');
add_line(model, 'WaitingRoom/1', 'Capture/1',     'autorouting','on');
add_line(model, 'Capture/1',     'QualityGate/1', 'autorouting','on');
add_line(model, 'QualityGate/1', 'Screened/1',    'autorouting','on');
% retake branch: second output of the switch goes back to the queue
try
    add_line(model, 'QualityGate/2', 'WaitingRoom/1', 'autorouting','on');
catch
    warning(['Could not auto-wire the retake branch. Open the model and ' ...
             'drag QualityGate output 2 back to WaitingRoom input.']);
end

% ---- parameters (best effort) ---------------------------------------
trySet([model '/Arrivals'],   'GenerationMethod',       'Time-based');
trySet([model '/Arrivals'],   'InterGenerationTimeSource','Dialog');
trySet([model '/Arrivals'],   'DistributionType',       'Exponential');
trySet([model '/Arrivals'],   'Mean',  num2str(P.mean_interarrival_min));

trySet([model '/WaitingRoom'],'Capacity', num2str(P.queue_capacity));
trySet([model '/WaitingRoom'],'StatisticsAverageWait','on');
trySet([model '/WaitingRoom'],'StatisticsNumberInBlock','on');

trySet([model '/Capture'],    'ServiceTimeSource','Dialog');
trySet([model '/Capture'],    'ServiceTime', num2str(P.capture_mean_min));
trySet([model '/Capture'],    'Capacity', num2str(P.n_capture_stations));
trySet([model '/Capture'],    'StatisticsUtilization','on');
trySet([model '/Capture'],    'StatisticsNumberDeparted','on');

trySet([model '/QualityGate'],'SwitchingCriterion','Equiprobable');
trySet([model '/QualityGate'],'NumberOfOutputPorts','2');

save_system(model);

fprintf('\n=====================================================\n');
fprintf('Model built and saved as %s.slx\n', model);
fprintf('Stop time  : %g minutes (one clinic day)\n', P.day_minutes);
fprintf('Retake rate: %.0f%% (set on the QualityGate block)\n', P.retake_prob*100);
fprintf('\nNext: press Run, then open the Data Inspector.\n');
fprintf('Any warnings above list blocks to set by hand in the dialog.\n');
fprintf('=====================================================\n');


function trySet(blk, param, value)
    try
        set_param(blk, param, value);
    catch
        [~, name] = fileparts(blk);
        fprintf('  [skip] %-14s -> %s (set this one in the block dialog)\n', ...
                name, param);
    end
end
