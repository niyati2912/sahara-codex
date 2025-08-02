document.getElementById("sosForm").addEventListener("submit", function(e) {
e.preventDefault();
const name = document.getElementById("seeker-name").value;
document.getElementById("sos-response").innerText = `🚨 SOS sent! Stay strong, ${name}. A helper will reach you soon.`;
});

document.getElementById("volunteerForm").addEventListener("submit", function(e) {
e.preventDefault();
const name = document.getElementById("volunteer-name").value;
document.getElementById("volunteer-response").innerText = `🙌 Thanks, ${name}! You're registered as a volunteer.`;
});
