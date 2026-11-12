"""
recon_lambda_test.py - quick visual check of the sparse-prior weight (lambda).

Reconstructs a small region of blocks for a few images at several lambda
values and saves stitched results side by side. Uses the same settings,
block geometry and preprocessing as isetbio_recon_piecewise_parallel.py
(imported for its settings only; its reconstruction does not run and its
output folders are not touched).

Output: ../stimulusSets/isettreeshrew/treeshrew_30/lambda_test/<set>_<category>/
    <image>_compare.png   original region | one panel per lambda
    all_compare.png       every image (rows) x lambda (columns)
    lambda_<val>/<image>  stitched reconstruction of the region
    stats.csv             per block/lambda: objective, data misfit, prior term
"""
import os, time, csv
import numpy as np, torch, h5py, scipy.io, cv2
import isetbio_recon_piecewise_parallel as base     # settings + block_geometry only
from inverse.bayes import SparseEstimator, BayesEstimator
from utils.dataset import gamma_correct

# ---------------- test settings ----------------
lambdas      = [1e-2, 2.5e-2, 5e-2, 1e-1]           # 1e-2 = value in the main script
runName      = 'round2'                             # output subfolder (round 1 = 1e-3,3e-3,1e-2,3e-2,1e-1)
block_rows   = range(2, 6)                          # 1-based block indices (as in MATLAB)
block_cols   = range(2, 8)
imageSetName = 'Camel_v2_test_nn'
category     = 'original'
images       = ['camel_55.png', 'camel_74.png', 'camel_77.png', 'wrench_0.png', 'wrench_92.png']
num_it       = base.num_it                          # same iterations as the main script
seed         = 0                                    # same random start for every lambda
# ------------------------------------------------

B, L, imOrig = base.imBorder, base.blockLen, base.imOrig
blockSize = base.blockSize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
outDir = (f'../stimulusSets/isettreeshrew/{base.species}_{base.sceneFOVdegs}/lambda_test/'
          f'{imageSetName}_{category}_{runName}')
os.makedirs(outDir, exist_ok=True)
lam_str = lambda l: f'{l:g}'

# ---- load and pad images exactly as process_block does (RGB path) ----
padded = []
for name in images:
    img = cv2.imread(f'../stimulusSets/{imageSetName}/{category}/{name}')
    img = cv2.resize(img, (imOrig, imOrig))
    vals, counts = np.unique(img, return_counts=True)
    bv = int(vals[np.argmax(counts)])
    rep = cv2.copyMakeBorder(img, B, B+1, B, B+1, borderType=cv2.BORDER_CONSTANT, value=(bv, bv, bv))
    padded.append((rep / 255.0).astype(np.float32))
padded = np.stack(padded)                           # (nImg, 279, 279, 3)

# region of the original image covered by the selected block cores
r0 = (block_rows[0]-1)*L; r1 = block_rows[-1]*L
c0 = (block_cols[0]-1)*L; c1 = block_cols[-1]*L
canvas = {l: np.zeros((len(images), r1-r0, c1-c0, 3), np.float32) for l in lambdas}

prior = scipy.io.loadmat('./assets/sparsePrior.mat')
basis = np.linalg.inv(prior['regBasis'])

stats = []
t_all = time.time()
for row_idx in block_rows:
    for col_idx in block_cols:
        t0 = time.time()
        row_start, col_start, ex, ey = base.block_geometry(row_idx, col_idx)
        f = f'{base.renderLoadPath}{base.species}_blocked/render_{base.sceneFOVdegs}_Xecc{ex}_Yecc{ey}.mat'
        with h5py.File(f, 'r') as h:
            render = np.array(h['renderMtx']).astype(np.float32)
        mtx = torch.tensor(render.T)
        sparse = SparseEstimator(device, render.T, basis, prior['mu'].T, lbda=lambdas[0])

        crop = padded[:, row_start:row_start+blockSize[0], col_start:col_start+blockSize[1], :]
        img_tor = torch.tensor(crop).permute(0, 3, 1, 2).contiguous()   # (nImg, 3, 55, 55)
        msmt = BayesEstimator.measure(mtx, img_tor).to(device)

        for l in lambdas:
            sparse.lbda = l
            torch.manual_seed(seed)
            rec, loss = sparse.recon(msmt, tuple(img_tor.shape), n_iter=num_it, print_loss=False)
            with torch.no_grad():
                misfit = sparse.neg_llhd(msmt, rec)
                pr = sparse.conv_prior(rec)
                rel = (torch.linalg.norm(BayesEstimator.measure(sparse.render, rec) - msmt, dim=1)
                       / torch.linalg.norm(msmt, dim=1))
            rec = rec.permute(0, 2, 3, 1).cpu().numpy()
            rr = (row_idx-1)*L - r0; cc = (col_idx-1)*L - c0
            for i in range(len(images)):
                canvas[l][i, rr:rr+L, cc:cc+L] = gamma_correct(rec[i])[B:B+L, B:B+L]
                stats.append([row_idx, col_idx, ex, ey, lam_str(l), images[i], loss[-1] / len(images),
                              misfit[i].item(), pr[i].item(), rel[i].item(), loss[0] / len(images)])
        print(f'block ({row_idx},{col_idx}) Xecc{ex} Yecc{ey}: {time.time()-t0:.1f} s', flush=True)

# ---------------- save ----------------
with open(f'{outDir}/stats.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['row','col','eccX','eccY','lambda','image','final_objective_mean','data_misfit',
                'prior_term','relative_residual','initial_objective_mean'])
    w.writerows(stats)

def label(im, text):
    im = cv2.copyMakeBorder(im, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(im, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return im

scale = 2
up = lambda a: cv2.resize(np.uint8(np.clip(a, 0, 1)*255), None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
gap = lambda h: np.full((h, 6, 3), 255, np.uint8)
rows_out = []
for i, name in enumerate(images):
    orig = padded[i, B+r0:B+r1, B+c0:B+c1]          # same pixels, original (non-reconstructed)
    panels = [label(up(orig), 'original')]
    for l in lambdas:
        d = f'{outDir}/lambda_{lam_str(l)}'
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(f'{d}/{name}', np.uint8(np.clip(canvas[l][i], 0, 1)*255))
        panels.append(label(up(canvas[l][i]), f'lambda {lam_str(l)}'))
    h = panels[0].shape[0]
    strip = np.concatenate(sum([[p, gap(h)] for p in panels], [])[:-1], axis=1)
    cv2.imwrite(f'{outDir}/{name[:-4]}_compare.png', strip)
    rows_out.append(label(strip, name))
cv2.imwrite(f'{outDir}/all_compare.png', np.concatenate(rows_out, axis=0))

print(f'done in {(time.time()-t_all)/60:.1f} min; saved to {os.path.abspath(outDir)}')
