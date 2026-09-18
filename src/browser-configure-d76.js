// paper18 browser-configure-d76.js -- configure query parameters in the WONDER UCD (D76) form
// Differences vs CMF D140: O_ucd (not O_icd); years via the F_D76.V1 multi-select (not V_D76.V1);
// B_2=D76.V1-level1 (not D76.V1); the finder also has V25/V27.
// The caller injects the years via window.PAPER18_YEARS = ["2017","2018"]
(() => {
  const f = document.forms['form'];
  const set = (n, v) => { const e = f.elements[n]; if (!e) return 'MISS:' + n + ';'; e.value = v; return ''; };
  let miss = '';
  miss += set('B_1', 'D76.V9-level1');      // State
  miss += set('B_2', 'D76.V1-level1');      // Year
  miss += set('B_3', 'D76.V2-level1');             // ICD Chapter (flat, mutually exclusive)
  miss += set('B_4', '*None*');
  miss += set('B_5', '*None*');
  const years = window.PAPER18_YEARS || ['2017', '2018'];
  const ysel = f.elements['F_D76.V1'];
  [...ysel.options].forEach(o => o.selected = years.includes(o.value));
  for (const n of ['F_D76.V2', 'F_D76.V4', 'F_D76.V9', 'F_D76.V10', 'F_D76.V22', 'F_D76.V25', 'F_D76.V27']) {
    const e = f.elements[n];
    if (e) { e.value = '*All*'; }
  }
  miss += set('O_title', 'paper18 ucd');
  miss += set('O_timeout', '900');
  miss += set('O_export-format', 'tsv');
  // AAR: enable the age-adjusted rate (2000 US standard population) + the M_9 measure
  const aarEn = [...f.querySelectorAll('input[name=O_aar_enable]')];
  aarEn.forEach(c => c.checked = true);
  const m9 = [...f.querySelectorAll('input[name="M_9"]')].find(c => c.value === 'D76.M9');
  if (m9) m9.checked = true;
  const ucd = [...f.querySelectorAll('input[name=O_ucd]')].find(r => r.value === 'D76.V4');
  ucd.checked = true;
  [...f.querySelectorAll('input[name=O_show_totals]')].forEach(c => c.checked = false);
  [...f.querySelectorAll('input[name=O_show_suppressed]')].forEach(c => c.checked = true);
  [...f.querySelectorAll('input[name=O_show_zeros]')].forEach(c => c.checked = true);
  return JSON.stringify({ miss, years: ysel.selectedOptions.length });
})()
