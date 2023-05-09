function dataL = GroupTDDataByLabel (data, ltype, boolCol2use, hemi, ophemi, varargin)

uo = {'uniformoutput', 0};

% re-organize based on indicated label type
switch lower(ltype)
    case 'medlev'
        for iH = 1:length(hemi)
            meds = unique(data.(hemi{iH}).MedState);
            
            for iM = 1:length(meds)
                boolMed = data.(hemi{iH}).MedState == meds(iM);
                dataL.(hemi{iH}){iM} = data.(hemi{iH})(boolMed, boolCol2use);
                fprintf('.');
            end
        end
    case 'apple'
        sympt = varargin{1};
        thresh = varargin{2};
        for iH = 1:length(hemi)
            fprintf('.');
            
            thresh.(hemi{iH}) = [0, thresh.(hemi{iH}), 1];
            for iT = 1:(length(thresh.(hemi{iH}))-1)
                fprintf('.');
                boolS = contains(data.(hemi{iH}).Properties.VariableNames, 'aw', 'ignorecase', 1) & ...
                    contains(data.(hemi{iH}).Properties.VariableNames, sympt.(hemi{iH}), 'ignorecase', 1);
                data.(hemi{iH}){(data.(hemi{iH}){:,boolS} > 1), boolS} = 1; % sometimes max is a little above 1
                if iT < (length(thresh.(hemi{iH}))-1)
                    boolInc = data.(hemi{iH}){:,boolS} >= thresh.(hemi{iH})(iT) & data.(hemi{iH}){:,boolS} < thresh.(hemi{iH})(iT+1);
                else
                    boolInc = data.(hemi{iH}){:,boolS} >= thresh.(hemi{iH})(iT) & data.(hemi{iH}){:,boolS} <= thresh.(hemi{iH})(iT+1);
                end
                dataL.(hemi{iH}){iT} = data.(hemi{iH})(boolInc, boolCol2use);
                
            end
            
            if sum(cell2mat(cellfun(@(x)(size(x,1)), dataL.(hemi{iH}), uo{:}))==0)
                disp(' ');
                error(['Threshold for (' hemi{iH} ') hemisphere does produces a class with no data points.']);
            end
        end
    case 'motordiary'
        sympt = varargin{1};
        thresh = varargin{2};
        symRange = varargin{3};
        for iH = 1:length(hemi)
            fprintf('.');
            
            thresh.(hemi{iH}) = [symRange.(hemi{iH})(1), thresh.(hemi{iH}), symRange.(hemi{iH})(2)];
            for iT = 1:(length(thresh.(hemi{iH}))-1)
                fprintf('.');
                boolS = contains(data.(hemi{iH}).Properties.VariableNames, 'md', 'ignorecase', 1) & ...
                    contains(data.(hemi{iH}).Properties.VariableNames, sympt.(hemi{iH}), 'ignorecase', 1);
                if iT < (length(thresh.(hemi{iH}))-1)
                    boolInc = data.(hemi{iH}){:,boolS} >= thresh.(hemi{iH})(iT) & data.(hemi{iH}){:,boolS} < thresh.(hemi{iH})(iT+1);
                else
                    boolInc = data.(hemi{iH}){:,boolS} >= thresh.(hemi{iH})(iT) & data.(hemi{iH}){:,boolS} <= thresh.(hemi{iH})(iT+1);
                end
                dataL.(hemi{iH}){iT} = data.(hemi{iH})(logical(sum(boolInc,2)), boolCol2use);
                
            end
            
            if sum(cell2mat(cellfun(@(x)(size(x,1)), dataL.(hemi{iH}), uo{:}))==0)
                disp(' ');
                error(['Threshold for (' hemi{iH} ') hemisphere does produces a class with no data points.']);
            end
        end
        
    case 'pkg'
        sympt = varargin{1};
        thresh = varargin{2};
        symRange = varargin{3};
        for iH = 1:length(hemi)
            fprintf('.');
            
            thresh.(hemi{iH}) = [symRange.(hemi{iH})(1), thresh.(hemi{iH}), symRange.(hemi{iH})(2)];
            for iT = 1:(length(thresh.(hemi{iH}))-1)
                fprintf('.');
                boolS = contains(data.(hemi{iH}).Properties.VariableNames, 'pkg', 'ignorecase', 1) & ...
                    contains(data.(hemi{iH}).Properties.VariableNames, sympt.(hemi{iH}), 'ignorecase', 1);
                if iT < (length(thresh.(hemi{iH}))-1)
                    boolInc = data.(hemi{iH}){:,boolS} >= thresh.(hemi{iH})(iT) & data.(hemi{iH}){:,boolS} < thresh.(hemi{iH})(iT+1);
                else
                    boolInc = data.(hemi{iH}){:,boolS} >= thresh.(hemi{iH})(iT) & data.(hemi{iH}){:,boolS} <= thresh.(hemi{iH})(iT+1);
                end
                dataL.(hemi{iH}){iT} = data.(hemi{iH})(boolInc, boolCol2use);
                
            end
            
            if sum(cell2mat(cellfun(@(x)(size(x,1)), dataL.(hemi{iH}), uo{:}))==0)
                disp(' ');
                error(['Threshold for (' hemi{iH} ') hemisphere does produces a class with no data points.']);
            end
        end
end