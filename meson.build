project('EdgeMasks', 'cpp',
  default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++17'],
  license: 'MIT',
  meson_version: '>=1.2.3',
  version: '3.2',
)

incdir = include_directories(run_command(find_program('python', 'python3'), '-c', 'import vapoursynth as vs; print(vs.get_include())', check: true).stdout().strip())

libs = []

py = import('python').find_installation(pure: false)

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-DEDGEMASKS_X86', language: 'cpp')

  libs += static_library('instrset_detect',
    files('EdgeMasks/vectorclass/instrset_detect.cpp'),
    gnu_symbol_visibility: 'hidden',
  )

  libs += static_library('sse4',
    files('EdgeMasks/edgemasks_sse4.cpp'),
    cpp_args: '-march=x86-64-v2',
    gnu_symbol_visibility: 'hidden',
    include_directories: incdir,
  )

  libs += static_library('avx2',
    files('EdgeMasks/edgemasks_avx2.cpp'),
    cpp_args: '-march=x86-64-v3',
    gnu_symbol_visibility: 'hidden',
    include_directories: incdir,
  )

  libs += static_library('avx512',
    files('EdgeMasks/edgemasks_avx512.cpp'),
    cpp_args: '-march=x86-64-v4',
    gnu_symbol_visibility: 'hidden',
    include_directories: incdir,
  )
endif

shared_module('edgemasks',
  files('EdgeMasks/edgemasks.cpp'),
  gnu_symbol_visibility: 'hidden',
  include_directories: incdir,
  install: true,
  install_dir: py.get_install_dir() / 'vapoursynth/plugins',
  link_args: host_machine.system() in ['cygwin', 'windows'] ? '-static' : [],
  link_with: libs,
  name_prefix: '',
)
