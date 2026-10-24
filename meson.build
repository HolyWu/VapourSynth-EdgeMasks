project('EdgeMasks', 'cpp',
    default_options: ['buildtype=release', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++20'],
    license: 'Apache-2.0 AND MIT',
    license_files: ['src/vectorclass/LICENSE', 'LICENSE'],
    meson_version: '>=1.6.0',
    version: '4.1',
)

py = import('python').find_installation(pure: false)

incdir = include_directories(
    run_command(py, '-c', 'import vapoursynth as vs; print(vs.get_include())', check: true).stdout().strip(),
)

libs = []

version = meson.project_version().split('.')

add_project_arguments(
    '-DEDGEMASKS_VERSION_MAJOR=@0@'.format(version[0].to_int()),
    '-DEDGEMASKS_VERSION_MINOR=@0@'.format(version.get(1, '0').to_int()),
    '-DVS_USE_API_42',
    '-fno-math-errno',
    '-fno-trapping-math',
    language: 'cpp',
)

if host_machine.cpu_family() == 'x86_64'
    add_project_arguments('-DEDGEMASKS_X86', language: 'cpp')

    libs += static_library('instrset_detect',
        files('src/vectorclass/instrset_detect.cpp'),
        gnu_symbol_visibility: 'hidden',
    )

    libs += static_library('sse4',
        files('src/edgemasks_sse4.cpp'),
        cpp_args: '-march=x86-64-v2',
        gnu_symbol_visibility: 'hidden',
        include_directories: incdir,
    )

    libs += static_library('avx2',
        files('src/edgemasks_avx2.cpp'),
        cpp_args: '-march=x86-64-v3',
        gnu_symbol_visibility: 'hidden',
        include_directories: incdir,
    )

    libs += static_library('avx512',
        files('src/edgemasks_avx512.cpp'),
        cpp_args: '-march=x86-64-v4',
        gnu_symbol_visibility: 'hidden',
        include_directories: incdir,
    )
endif

shared_module('edgemasks',
    files('src/edgemasks.cpp'),
    gnu_symbol_visibility: 'hidden',
    include_directories: incdir,
    install: true,
    install_dir: py.get_install_dir() / 'vapoursynth/plugins',
    link_with: libs,
    name_prefix: '',
)
