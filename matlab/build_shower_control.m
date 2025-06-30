function build_shower_control(init_state)
  mdl = 'shower_control';
  modelDir = fullfile(pwd,'models');
  if ~exist(modelDir,'dir'), mkdir(modelDir); end
  if bdIsLoaded(mdl), close_system(mdl,0); end
  if exist(fullfile(modelDir,[mdl '.slx']),'file')
    delete(fullfile(modelDir,[mdl '.slx']));
  end

  new_system(mdl); load_system('simulink');
  add_block('simulink/Sources/In1',      [mdl '/u'],      'Position',[50 50 80 80]);
  add_block('simulink/Discrete/Unit Delay',[mdl '/Delay'], 'Position',[150 50 180 80], ...
            'X0', sprintf('%d', init_state));
  add_block('simulink/Math Operations/Sum', [mdl '/Sum'],  'Inputs','++', 'Position',[100 100 130 130]);
  add_block('simulink/Sinks/Out1',         [mdl '/y'],      'Position',[250 80 280 110]);
  add_block('simulink/Sinks/To Workspace',[mdl '/ToWS'],   'VariableName','sim_state', ...
            'SaveFormat','Array','Position',[250 120 280 150]);

  add_line(mdl,'u/1','Sum/1');
  add_line(mdl,'Delay/1','Sum/2');
  add_line(mdl,'Sum/1','Delay/1');
  add_line(mdl,'Sum/1','y/1');
  add_line(mdl,'Sum/1','ToWS/1');

  save_system(mdl,fullfile(modelDir,[mdl '.slx']));
  close_system(mdl);
  fprintf('✓ Simulink-Modell %s gespeichert\n',mdl);
end
