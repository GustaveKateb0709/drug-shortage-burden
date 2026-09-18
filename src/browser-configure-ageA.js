// paper18 browser-configure.js -- configure query parameters in the WONDER CMF form
// Usage (a browser-automation CLI cannot load files directly with eval; this file is kept in src for
// reproducibility documentation, see README):
//   Configures the fields, serializes the FormData, and clicks Send automatically.
//   The caller injects the years via window.PAPER18_YEARS = ["2003","2004"]
(() => {
  const f = document.forms['form'];
  const set = (n, v) => { const e = f.elements[n]; if (!e) return 'MISS:' + n + ';'; e.value = v; return ''; };
  let miss = '';
  miss += set('B_1', 'D140.V9-level1');   // State
  miss += set('B_2', 'D140.V1');          // Year
  miss += set('B_3', 'D140.V5');   // ICD-10 113 Cause List
  miss += set('B_4', '*None*');
  miss += set('B_5', '*None*');
  const years = window.PAPER18_YEARS || ['2003', '2004'];
  [...f.elements['V_D140.V1'].options].forEach(o => o.selected = years.includes(o.value));
  miss += set('F_D140.V2', 'A00-B99');
  miss += set('F_D140.V4', '*All*');
  miss += set('F_D140.V9', '*All*');
  miss += set('F_D140.V10', '*All*');
  miss += set('F_D140.V18', '*All*');
  miss += set('O_title', 'paper18');
  miss += set('O_timeout', '900');
  miss += set('O_export-format', 'tsv');
  // Note: WONDER does not allow AAR output when grouping by age (verified empirically), so O_aar_enable/M_9 are not enabled
  const icd = [...f.querySelectorAll('input[name=O_icd]')].find(r => r.value === 'D140.V2');
  icd.checked = true;
  [...f.querySelectorAll('input[name=O_show_totals]')].forEach(c => c.checked = false);
  [...f.querySelectorAll('input[name=O_show_suppressed]')].forEach(c => c.checked = true);
  [...f.querySelectorAll('input[name=O_show_zeros]')].forEach(c => c.checked = true);
  return JSON.stringify({ miss, years: f.elements['V_D140.V1'].selectedOptions.length });
})()
