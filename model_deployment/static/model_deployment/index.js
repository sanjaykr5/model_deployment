function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

document.addEventListener('DOMContentLoaded', function () {

    // Event listener for main form submission
    var inputForm = document.getElementById('inputForm');
    if (inputForm) {
        inputForm.addEventListener('submit', function (event) {
            event.preventDefault();

            var formData = new FormData(inputForm);

            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/submit_form', true);  // Adjust URL as per your Django view
            xhr.setRequestHeader('X-CSRFToken', getCookie('csrftoken'));

            xhr.onload = function () {
                if (xhr.status >= 200 && xhr.status < 300) {
                    var response = JSON.parse(xhr.responseText);
                    document.getElementById('output1').value = response['product_title'];
                    document.getElementById('output2').value = response['class_pred'];
                    document.getElementById('output3').value = response['clus_pred'];
                } else {
                    console.error('Request failed with status:', xhr.status, xhr.statusText);
                }
            };

            xhr.onerror = function () {
                console.error('Request failed');
            };

            xhr.send(formData);
        });
    }
});

document.addEventListener('DOMContentLoaded', function () {
    var classificationModelDropdown = document.getElementById('productClassificationModel');
    var clusteringModelDropdown = document.getElementById('productClusteringModel');
    classificationModelDropdown.addEventListener('change', handleDropdownChange);
    clusteringModelDropdown.addEventListener('change', handleDropdownChange);

    function handleDropdownChange(event) {
        let json_data = JSON.stringify({
            'classification_model': classificationModelDropdown.selectedIndex,
            'clustering_model': clusteringModelDropdown.selectedIndex
        });
        console.log(json_data)
        fetch('/model_selection', {
            method: 'POST',
            body: json_data,
            headers: {
                'X-CSRFToken': getCookie('csrftoken')
            }
        }).then({});
    }
});