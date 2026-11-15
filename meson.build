project('EdgeMasks', 'cpp',
    default_options: ['buildtype=release', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++20'],
    license: 'Apache-2.0 AND MIT',
    license_files: ['src/vectorclass/LICENSE', 'LICENSE'],
    meson_version: '>=1.6.0',
    version: '4.1',
)

libs = []

py = import('python').find_installation(pure: false)

# Older headers give VS_USE_API_43 the deprecated VSColorRange, whose VSC_RANGE_FULL is 0.
vs_release = run_command(
    py, '-c', 'import vapoursynth as vs; print(vs.__version__.release_major)',
    check: true,
).stdout().strip().to_int()

if vs_release < 82
    error('VapourSynth R82 or later is required, but R@0@ was found'.format(vs_release))
endif

version = meson.project_version().split('.')

vulkan_dep = dependency('vulkan',
    fallback: ['vulkan-headers', 'vulkan_headers_dep'],
    version: '>=1.4.304',
).partial_dependency(compile_args: true, includes: true)

add_project_arguments(
    '-DEDGEMASKS_VERSION_MAJOR=@0@'.format(version[0].to_int()),
    '-DEDGEMASKS_VERSION_MINOR=@0@'.format(version.get(1, '0').to_int()),
    '-DVS_USE_API_43',
    '-ffp-contract=fast',
    '-fno-math-errno',
    '-fno-trapping-math',
    '-I' + run_command(py, '-c', 'import vapoursynth as vs; print(vs.get_include())', check: true).stdout().strip(),
    language: 'cpp',
)

if host_machine.cpu_family() == 'x86_64'
    add_project_arguments('-DEDGEMASKS_X86', language: 'cpp')

    libs += static_library('instrset_detect',
        files('src/vectorclass/instrset_detect.cpp'),
        gnu_symbol_visibility: 'hidden',
    )

    foreach name, args : {
        'sse4': ['-march=x86-64-v2'],
        'avx2': ['-march=x86-64-v3', '-mstackrealign'],
        'avx512': ['-march=x86-64-v4', '-mstackrealign'],
    }
        libs += static_library(name,
            files('src/edgemasks_' + name + '.cpp'),
            cpp_args: args,
            gnu_symbol_visibility: 'hidden',
        )
    endforeach
endif

shared_module('edgemasks',
    files('src/edgemasks.cpp', 'src/edgemasks_vulkan.cpp'),
    dependencies: vulkan_dep,
    gnu_symbol_visibility: 'hidden',
    install: true,
    install_dir: py.get_install_dir() / 'vapoursynth/plugins',
    link_with: libs,
    name_prefix: '',
)
