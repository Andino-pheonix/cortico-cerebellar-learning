const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const root = path.resolve(__dirname, '..');
const directories = [
  'experiments/distributed/figures',
  'experiments/distributed/tuning/figures'
];
(async () => {
  for (const relative of directories) {
    const directory = path.join(root, relative);
    for (const file of fs.readdirSync(directory).filter(x => x.endsWith('.svg'))) {
      await sharp(path.join(directory, file), {density: 144})
        .resize({width: 1800}).png()
        .toFile(path.join(directory, file.replace(/\.svg$/, '.png')));
      process.stdout.write(path.join(relative, file) + ' rendered\n');
    }
  }
})().catch(error => { process.stderr.write(error.message + '\n'); process.exitCode = 1; });
