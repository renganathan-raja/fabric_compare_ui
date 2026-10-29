// Google Apps Script that stores the app's votes in the Google Sheet it is attached to.
// Setup: see "Host on Streamlit Community Cloud" in README.md.
// Script property SECRET must equal votes_webhook_secret in the Streamlit secrets.

function check_(secret) {
  return secret === PropertiesService.getScriptProperties().getProperty("SECRET");
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function doPost(e) {
  var body = JSON.parse(e.postData.contents);
  if (!check_(body.secret)) return json_({error: "bad secret"});
  var lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
    if (sheet.getLastRow() === 0) sheet.appendRow(body.fields);
    else {
      // columns added later (e.g. attribute_embedding) get their header on first use
      var header = sheet.getRange(1, 1, 1, Math.max(sheet.getLastColumn(), 1)).getValues()[0];
      for (var i = header.length; i < body.fields.length; i++) sheet.getRange(1, i + 1).setValue(body.fields[i]);
    }
    sheet.appendRow(body.row);
  } finally {
    lock.releaseLock();
  }
  return json_({ok: true});
}

function doGet(e) {
  if (!check_(e.parameter.secret)) return json_({error: "bad secret"});
  var values = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0].getDataRange().getDisplayValues();
  if (values.length < 2) return json_([]);
  var header = values[0];
  return json_(values.slice(1).map(function (row) {
    var out = {};
    header.forEach(function (name, i) { out[name] = row[i]; });
    return out;
  }));
}
