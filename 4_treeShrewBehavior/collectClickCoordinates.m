function [xCoords, yCoords] = collectClickCoordinates(dlcFrame_side)
% COLLECTCLICKCOORDINATES Display the middle frame of each trial and collect
% a single clicked (x, y) coordinate per trial.
%
% INPUT:
%   dlcFrame_side : cell array of trials, where each cell contains a 4D
%                   array of size (H x W x C x T)
%
% OUTPUT:
%   xCoords       : numeric vector of x coordinates, one per trial
%   yCoords       : numeric vector of y coordinates, one per trial
%
% USAGE:
%   [xCoords, yCoords] = collectClickCoordinates(dlcFrame_side);

    nTrials = numel(dlcFrame_side);

    xCoords = nan(1, nTrials);
    yCoords = nan(1, nTrials);

    % Create a single figure to reuse across trials
    fig = figure('Name', 'Click a Point', 'NumberTitle', 'off');

    for trial = 1:nTrials

        if ~isempty(dlcFrame_side)

            % Extract the middle frame for this trial
            nFrames    = size(dlcFrame_side{trial}, 4);
            midFrame   = round(nFrames / 2);
            frameImage = dlcFrame_side{trial}(:, :, :, midFrame);
    
            % Display the frame
            imshow(frameImage, []);
            title(sprintf('Trial %d / %d    Click one point, then press Enter', ...
                           trial, nTrials), 'FontSize', 13);
            drawnow;
    
            % Collect exactly one click (press Enter / any key to confirm)
            [x, y] = ginput(1);
    
            % Guard against the user closing the figure mid-session
            if ~ishandle(fig)
                warning('Figure was closed at trial %d. Stopping early.', trial);
                break;
            end
    
            xCoords(trial) = x;
            yCoords(trial) = y;
    
            % Overlay the clicked point so the user gets visual feedback
            hold on;
            plot(x, y, 'r+', 'MarkerSize', 14, 'LineWidth', 2);
            plot(x, y, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
            hold off;
            drawnow;
            pause(0.4);   % brief pause so the marker is visible before moving on
        end

    end

    close(fig);

    % -- Summary --------------------------------------------------------------
    fprintf('\nCoordinates collected for %d trial(s):\n', nTrials);
    fprintf('  %-8s %-10s %-10s\n', 'Trial', 'X', 'Y');
    fprintf('  %-8s %-10s %-10s\n', '-----', '---', '---');
    for k = 1:nTrials
        fprintf('  %-8d %-10.2f %-10.2f\n', k, xCoords(k), yCoords(k));
    end
end