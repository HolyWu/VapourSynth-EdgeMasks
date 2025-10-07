project('EdgeMasks', 'cpp',
  default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++17'],
  license: 'MIT',
  meson_version: '>=0.51.0',
  version: '3.1',
)

cxx = meson.get_compiler('cpp')

gcc_syntax = cxx.get_argument_syntax() == 'gcc'

if get_option('buildtype') == 'release'
  add_project_arguments(gcc_syntax ? ['-fno-math-errno', '-fno-trapping-math'] : '/GS-', language: 'cpp')
endif

sources = files('EdgeMasks/edgemasks.cpp')

if gcc_syntax
  vapoursynth_dep = dependency('vapoursynth', version: '>=55').partial_dependency(compile_args: true, includes: true)
  install_dir = vapoursynth_dep.get_variable(pkgconfig: 'libdir') / 'vapoursynth'
else
  vapoursynth_dep = []
  install_dir = get_option('libdir') / 'vapoursynth'
endif

libs = []

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-DEDGEMASKS_X86', language: 'cpp')

  libs += static_library('instrset_detect',
    files('EdgeMasks/vectorclass/instrset_detect.cpp'),
    gnu_symbol_visibility: 'hidden',
  )

  libs += static_library('sse4',
    files('EdgeMasks/edgemasks_sse4.cpp'),
    cpp_args: gcc_syntax ? '-march=x86-64-v2' : '/arch:SSE2',
    dependencies: vapoursynth_dep,
    gnu_symbol_visibility: 'hidden',
  )

  libs += static_library('avx2',
    files('EdgeMasks/edgemasks_avx2.cpp'),
    cpp_args: gcc_syntax ? '-march=x86-64-v3' : '/arch:AVX2',
    dependencies: vapoursynth_dep,
    gnu_symbol_visibility: 'hidden',
  )

  libs += static_library('avx512',
    files('EdgeMasks/edgemasks_avx512.cpp'),
    cpp_args: gcc_syntax ? '-march=x86-64-v4' : '/arch:AVX512',
    dependencies: vapoursynth_dep,
    gnu_symbol_visibility: 'hidden',
  )
endif

shared_module('edgemasks',
  sources,
  dependencies: vapoursynth_dep,
  gnu_symbol_visibility: 'hidden',
  install: true,
  install_dir: install_dir,
  link_with: libs,
  name_prefix: '',
)
