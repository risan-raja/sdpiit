call plug#begin()
if has('nvim')
  Plug 'Shougo/deoplete.nvim', { 'do': ':UpdateRemotePlugins' }
else
  Plug 'Shougo/deoplete.nvim'
  Plug 'roxma/nvim-yarp'
  Plug 'roxma/vim-hug-neovim-rpc'
endif
"Plug 'Shougo/ddc.vim'
"Plug 'vim-denops/denops.vim'
"Plug 'LumaKernel/ddc-tabnine'
" Install your sources
"Plug 'Shougo/ddc-around'

" Install your filters
"Plug 'Shougo/ddc-matcher_head'
"Plug 'Shougo/ddc-sorter_rank'
  Plug 'neovim/nvim-lspconfig'
  Plug 'wbthomason/packer.nvim'
  Plug 'williamboman/nvim-lsp-installer'
if has('win32') || has('win64')
  Plug 'tbodt/deoplete-tabnine', { 'do': 'powershell.exe .\install.ps1' }
else
  Plug 'tbodt/deoplete-tabnine', { 'do': './install.sh' }
endif
call plug#end()
let g:deoplete#enable_at_startup = 0
let g:python3_host_prog='/home/mlop3n/temp2/bin/python3'
" Customize global settings
" Use around source.
" https://github.com/Shougo/ddc-around
"call ddc#custom#patch_global('sources', ['around','tabnine'])
"call ddc#custom#patch_global('sourceOptions', {
"    \ 'tabnine': {
"    \   'mark': 'TN',
"    \   'maxCandidates': 5,
"    \   'isVolatile': v:true,
"    \ }})
"
" Use matcher_head and sorter_rank.
" https://github.com/Shougo/ddc-matcher_head
" https://github.com/Shougo/ddc-sorter_rank
"call ddc#custom#patch_global('sourceOptions', {
"      \ '_': {
"      \   'matchers': ['matcher_head'],
"      \   'sorters': ['sorter_rank']},
"      \ })
"
"" Change source options
"call ddc#custom#patch_global('sourceOptions', {
"      \ 'around': {'mark': 'A'},
"      \ })
"call ddc#custom#patch_global('sourceParams', {
"      \ 'around': {'maxSize': 500},
"      \ })
"
"" Customize settings on a filetype
"call ddc#custom#patch_filetype(['c', 'cpp'], 'sources', ['around', 'clangd'])
"call ddc#custom#patch_filetype(['c', 'cpp'], 'sourceOptions', {
"      \ 'clangd': {'mark': 'C'},
""      \ })
"call ddc#custom#patch_filetype('markdown', 'sourceParams', {
"      \ 'around': {'maxSize': 100},
""      \ })
"
"" Mappings

"" <TAB>: completion.
inoremap <silent><expr> <TAB> pumvisible() ? "\<C-n>" : "\<TAB>"
"\ ddc#map#pum_visible() ? '<C-n>' :
"\ (col('.') <= 1 <Bar><Bar> getline('.')[col('.') - 2] =~# '\s') ?
"\ '<TAB>' : ddc#map#manual_complete()
"
"" <S-TAB>: completion back.
inoremap <silent><expr><S-TAB>  pum_visible() ? "<C-p>" : "\<TAB>"
"
" Use ddc.
"call ddc#enable()

"#call ddc#custom#patch_global('sources', ['nvim-lsp'])
"#call ddc#custom#patch_global('sourceOptions', {
"#      \ '_': { 'matchers': ['matcher_head'] },
"#      \ 'nvim-lsp': {
"#      \   'mark': 'lsp',
"#      \   'forceCompletionPattern': '\.\w*|:\w*|->\w*' },
"#      \ })

"" Use Customized labels
"#call ddc#custom#patch_global('sourceParams', {
"#      \ 'nvim-lsp': { 'kindLabels': { 'Class': 'c' } },
"#      \ })
